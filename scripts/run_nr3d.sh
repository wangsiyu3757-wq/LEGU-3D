#!/usr/bin/env bash
# NR3D grounding run (llm_grounding.py), detached so it survives a closed terminal.
# Re-running the same command resumes from <EXP>.ckpt.jsonl.
#
#   scripts/run_nr3d.sh                                   # quadrant, port 8000
#   SCHEME=compass_cutoff scripts/run_nr3d.sh
#   SCHEME=compass EXP=nr3d_compass LLM_URL=http://127.0.0.1:8001/v1/chat/completions scripts/run_nr3d.sh
#
# Overridable: SCHEME (quadrant|compass|compass_cutoff), EXP, LLM_URL, LLM_MODEL,
#              PROG, TOP_K, MAX_SAMPLES, PY

set -euo pipefail
cd "$(dirname "$0")/.."

SCHEME="${SCHEME:-quadrant}"
EXP="${EXP:-nr3d_${SCHEME}}"
LLM_URL="${LLM_URL:-http://127.0.0.1:8000/v1/chat/completions}"
LLM_MODEL="${LLM_MODEL:-Qwen/Qwen3-32B-AWQ}"
PROG="${PROG:-data/nr3d_qwen3_feats.json}"
TOP_K="${TOP_K:-15}"
PY="${PY:-python}"

if pgrep -f -- "--exp_name ${EXP}( |\$)" >/dev/null; then
  echo "[run_nr3d] a job named '${EXP}' is already running; not starting another." >&2
  exit 1
fi

if ! curl -s -m 5 -o /dev/null "${LLM_URL%/chat/completions}/models"; then
  echo "[run_nr3d] LLM server not reachable at ${LLM_URL}; start it first." >&2
  exit 1
fi

ARGS=(--prog_path "$PROG" --exp_name "$EXP" --direction-scheme "$SCHEME"
      --llm_url "$LLM_URL" --llm_model "$LLM_MODEL" --top_k "$TOP_K")
[[ -n "${MAX_SAMPLES:-}" ]] && ARGS+=(--max_samples "$MAX_SAMPLES")
[[ "${NO_DCR:-0}" == 1 ]] && ARGS+=(--no_dcr)
[[ "${TARGET_TRACE:-0}" == 1 ]] && ARGS+=(--target_trace)

# Labels come straight from feats_3d.pkl, so grounding needs no GPU;
# keep it off the GPU to avoid contending with the vLLM server.
CUDA_DEVICE="${NR3D_GPU:-}"

if [[ "${FOREGROUND:-0}" == 1 ]]; then   # run in the foreground (e.g. inside a Slurm job)
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "$PY" llm_grounding.py "${ARGS[@]}" >> "${EXP}.out" 2>&1
  exit $?
fi

CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" setsid nohup "$PY" llm_grounding.py "${ARGS[@]}" \
  >> "${EXP}.out" 2>&1 < /dev/null &
disown
echo "[run_nr3d] started '${EXP}' (scheme=${SCHEME}) pid $!"
echo "  progress: tail -c 300 ${EXP}.out"
echo "  results:  ${EXP}.log  (written only when the run finishes)"
