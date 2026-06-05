<p align="center">
  <h1 align="center">LEGO-3D:LLM-parsed Explicit Geometry for Open-vocabulary 3D Grounding</h1>
</p>

## Overview

3D instance segmentation in complex indoor environments is a fundamental challenge, with applications in embodied AI, robotics, and augmented reality. Existing methods are constrained by closed-set category recognition and imprecise spatial delineation, fundamentally limiting generalization. 

We propose a geometry-aware zero-shot 3D instance segmentation framework with three core contributions:
1) LLM-based Token Structuring (LTS) (Stage 1/2): A prompted Qwen LLM decomposes the full query into a structured slot dictionary, assigning each entity a BOX label (BOX0 = target, BOX1/BOX2 = anchors), correctly handling long noun phrases, conjunctions, and embedded
attribute clauses.
2) Row-by-Row Object Listing (ROL) (Stage 3): We introduce ROL, which converts candidate objects into a structured attribute table for LLM-based grounding. Each row encodes one instance with its Euclidean distance to the anchor, relative orientation, and full bounding-box scale, converting target selection into a direct table lookup task that requires no implicit geometric inference.
3) Dense Coordinate Rendering (DCR) (Stage 3): For each candidate, the full bounding-box physical scale size = (W×H×D) and Euclidean distance are serialised verbatim into the prompt. This geometric pre-processing makes the LLM behave like an “engineering-literate” agent that can autonomously weigh physical dimensions (large/small/tall/short) against spatial proximity (near/far/left/right) to resolve fine-grained referential ambiguity without any arithmetic computation.

Experiments on ScanRefer achieve **38.9%** overall accuracy under a zero-shot protocol—competitive with the ZSVG3D baseline—while the LLM backbone is migrated to a locally deployed Qwen model, removing all proprietary API dependencies.

---

## Instructions

### Environment
The code is tested on Ubuntu with the following key package dependencies:
```bash
python==3.9.12
pytorch==1.11.0
pytorch3d==0.7.2
langchain==0.2.1
vllm
```
1. Locally Deploying the LLM (via vLLM)
We use vllm to host the open-source Qwen model locally, eliminating external API dependencies. Start the local inference server with the following command:
code
```Bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-32B-AWQ \
  --gpu-memory-utilization 0.98 \
  --max-model-len 4096 \
  --max-num-seqs 1 \
  --enforce-eager \
  --port 8000
```
2. Zero-Shot Evaluation on ScanRefer
Step 2.1: Generate Query Programs
Generate the parsed queries and structured programs using the locally hosted LLM:
code
```Bash
python generate_scanrefer_baseline.py \
  --scanrefer_path data/scanrefer_val.json \
  --prompt_path prompts/prompt_n32_gemma.txt \
  --output data/scanrefer_programs_baseline_gemma.json \
  --llm_url http://127.0.0.1:8000/v1 \
  --llm_model google/gemma-4-E2B-it \
  --resume
```
Step 2.2: Run Grounding Evaluation
Evaluate the generated programs and output the performance metrics:
code
```Bash
python llm_grounding_scanrefer_test.py
```
3. Zero-Shot Evaluation on Nr3D
Step 3.1: Generate Query Programs
Generate parsed query programs for the Nr3D dataset:
code
```Bash
python gen_visprogapple.py
```
Step 3.2: Run Grounding Evaluation
Run the grounding evaluation script directly to compute final results:
code
```Bash
python3 -c "
from zsvg.llm_grounding import run_llm_grounding_eval
run_llm_grounding_eval(
    prog_path='data/nr3d_val.json',
    exp_name='llm_grounding_feats_20',
)
"
```
Data Preparation
Please refer to the dataset configuration files to preprocess the ScanNet 3D scans, 2D images, and instance masks, ensuring they are placed in the data/ directory before running the scripts.
code
Code
