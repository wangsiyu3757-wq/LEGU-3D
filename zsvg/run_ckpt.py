"""Resumable evaluation loop shared by the NR3D and ScanRefer grounding scripts.

Each finished example is appended to a JSONL checkpoint, so a killed run can be
restarted with the same command and picks up where it stopped. Examples whose LLM
call failed are never checkpointed (they are retried at the end and on any restart),
and the run aborts instead of silently scoring fallbacks if the LLM stays down.
"""

import json
import os
import sys

from tqdm import tqdm


def _load(path, signature):
    done = {}
    if not os.path.exists(path):
        return done
    with open(path) as f:
        for n, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # torn last line from a hard kill
            if n == 0 and "_signature" in rec:
                if rec["_signature"] != signature:
                    sys.exit(f"[ABORT] {path} was written with a different configuration:\n"
                             f"  checkpoint: {rec['_signature']}\n  current:    {signature}\n"
                             f"Use a different --exp_name or delete the checkpoint.")
                continue
            done[rec["i"]] = rec
    return done


def run_with_ckpt(programs, eval_one, ckpt_path, signature, desc,
                  max_consecutive_failures=5, retry_rounds=2):
    """eval_one(program) -> dict with counter deltas under "d" and "llm_ok" (bool).

    Returns (records in program order, number of examples still scored by fallback).
    """
    done = _load(ckpt_path, signature)
    if done:
        print(f"[INFO] resuming: {len(done)}/{len(programs)} examples loaded from {ckpt_path}")

    records, failed, streak = {}, [], 0
    fresh = not os.path.exists(ckpt_path)
    with open(ckpt_path, "a") as ckpt:
        if fresh:
            ckpt.write(json.dumps({"_signature": signature}) + "\n")

        def save(rec):
            ckpt.write(json.dumps(rec) + "\n")
            ckpt.flush()

        for i, prog in enumerate(tqdm(programs, desc=desc)):
            if i in done:
                records[i] = done[i]
                continue
            rec = eval_one(prog)
            rec["i"] = i
            records[i] = rec
            if rec["llm_ok"]:
                streak = 0
                save(rec)
            else:
                failed.append(i)
                streak += 1
                if streak >= max_consecutive_failures:
                    print(f"\n[ABORT] LLM call failed on {streak} consecutive examples (server down?). "
                          f"Progress is saved in {ckpt_path}; restart the same command to resume.")
                    sys.exit(3)

        for _ in range(retry_rounds):
            if not failed:
                break
            still = []
            for i in failed:
                rec = eval_one(programs[i])
                rec["i"] = i
                if rec["llm_ok"]:
                    records[i] = rec
                    save(rec)
                else:
                    still.append(i)
            failed = still

    return [records[i] for i in range(len(programs))], len(failed)


def sum_deltas(records):
    total = {}
    for rec in records:
        for k, v in rec["d"].items():
            total[k] = total.get(k, 0) + v
    return total
