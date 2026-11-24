import os
import re
import json
import time
import argparse
import numpy as np
import requests
from tqdm import tqdm
from typing import List, Dict, Optional


# =====================================================================
# 1. 数据加载
# =====================================================================

def load_scanrefer(path: str) -> List[Dict]:
    with open(path, "r") as f:
        data = json.load(f)
    print(f"✅ 加载 ScanRefer: {len(data)} 条样本")
    return data


# =====================================================================
# 2. Program 生成（Completions API + 强制续写）
# =====================================================================

def build_prompt(system_prompt: str, caption: str) -> str:
    return (
        f"{system_prompt}\n"
        f"Description: {caption}\n"
        "Program:\n"
        "BOX0=LOC"
    )


def postprocess_program(raw: str) -> str:
    full = "BOX0=LOC" + raw

    lines = []
    for line in full.split('\n'):
        line = line.strip()
        if not line:
            continue
        if line.startswith('#'):
            continue

        # 修复无 object= 的 LOC
        if 'LOC(' in line and 'object=' not in line:
            inner = line.split('LOC(', 1)[1].rstrip(')').strip().strip("'\"")
            inner = inner.replace('_', ' ').lower()
            prefix = line.split('LOC')[0]
            line = f"{prefix}LOC(object='{inner}')"

        if re.match(r'^(BOX[0-9]+|TARGET|CENTER)[ ]*=', line):
            lines.append(line)
            if line.startswith('TARGET='):
                break
        else:
            continue

    return '\n'.join(lines)


def generate_program(
    caption: str,
    system_prompt: str,
    llm_url: str,
    llm_model: str,
    max_retries: int = 3,
) -> str:
    url = f"{llm_url.rstrip('/')}/completions"
    headers = {
        "Authorization": f"Bearer {os.environ.get('VLLM_API_KEY', 'EMPTY')}",
        "Content-Type": "application/json",
    }
    prompt = build_prompt(system_prompt, caption)

    for attempt in range(max_retries):
        try:
            payload = {
                "model": llm_model,
                "prompt": prompt,
                "temperature": 0.0,
                "max_tokens": 1028,
                "stop": ["\n\n", "Description:", "# EXAMPLE", "# TASK"],
                "echo": False,
            }
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["text"].strip()
            program = postprocess_program(raw)

            if "BOX0=LOC" in program and "TARGET=" in program:
                return program

            if attempt == max_retries - 1:
                return program if program else f"ERROR: empty program. raw={raw[:200]}"

        except Exception as e:
            if attempt == max_retries - 1:
                return f"ERROR: {str(e)}"

        time.sleep(1)

    return "ERROR: max retries exceeded"


# =====================================================================
# 3. 主流程
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="ScanRefer Visual Program 生成器（基线，无距离增强）")
    parser.add_argument('--scanrefer_path', required=True,
                        help='ScanRefer val JSON 文件路径')
    parser.add_argument('--prompt_path', default='prompts/prompt_n32_gemma.txt',
                        help='System prompt 文件路径')
    parser.add_argument('--output', default='data/scanrefer_programs_baseline.json',
                        help='输出 JSON 文件路径')
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--save_interval', type=int, default=50)
    parser.add_argument('--llm_url', default='http://127.0.0.1:8000/v1')
    parser.add_argument('--llm_model', default='google/gemma-4-26B-A4B-it')
    parser.add_argument('--resume', action='store_true',
                        help='从已有输出文件断点续跑')
    args = parser.parse_args()

    # ── 加载 system prompt ──────────────────────────────────────────
    if not os.path.exists(args.prompt_path):
        print(f"❌ prompt 文件不存在: {args.prompt_path}")
        return
    with open(args.prompt_path, 'r') as f:
        system_prompt = f.read().strip()
    print(f"✅ 加载 prompt: {args.prompt_path} ({len(system_prompt)} chars)")

    # ── 加载数据 ────────────────────────────────────────────────────
    data = load_scanrefer(args.scanrefer_path)
    if args.max_samples:
        data = data[:args.max_samples]
    print(f"   处理样本数: {len(data)}")

    # ── 断点续跑 ────────────────────────────────────────────────────
    results = []
    done_keys = set()
    if args.resume and os.path.exists(args.output):
        with open(args.output, 'r') as f:
            results = json.load(f)
        done_keys = {
            f"{r['scan_id']}_{r['target_id']}_{r.get('ann_id', '')}"
            for r in results
        }
        print(f"♻️  续跑: 已完成 {len(done_keys)} 条")

    # ── 输出目录 ────────────────────────────────────────────────────
    os.makedirs(
        os.path.dirname(args.output) if os.path.dirname(args.output) else '.',
        exist_ok=True
    )
    tmp_path = args.output.replace('.json', '_temp.json')

    pbar = tqdm(total=len(data), desc="生成 programs")

    for idx, item in enumerate(data):
        pbar.update(1)

        scan_id   = item['scan_id']
        target_id = int(item['target_id'])
        ann_id    = item.get('ann_id', str(idx))
        caption   = item['caption']      # 原始 caption，不做任何增强

        dedup_key = f"{scan_id}_{target_id}_{ann_id}"
        if dedup_key in done_keys:
            continue

        # ── 生成 Program（直接用原始 caption）──────────────────────
        program = generate_program(
            caption=caption,
            system_prompt=system_prompt,
            llm_url=args.llm_url,
            llm_model=args.llm_model,
        )

        is_error = program.startswith("ERROR")

        result = {
            "scan_id":          scan_id,
            "caption":          caption,        # 原始 caption，无增强
            "caption_original": caption,
            "target_id":        str(target_id),
            "target_box":       None,
            "pred_box":         None,
            "pred_id":          None,
            "pred_name":        None,
            "iou":              0.0,
            "success":          None,
            "program":          program if not is_error else None,
            "error":            program if is_error else None,
            "ann_id":           ann_id,
            "object_name":      item.get('object_name', ''),
        }

        results.append(result)
        done_keys.add(dedup_key)

        status = "✅" if not is_error else "❌"
        tqdm.write(
            f"{status} [{idx+1}/{len(data)}] {scan_id} | tid={target_id} | ann={ann_id}"
        )
        if not is_error:
            tqdm.write(f"   {program.split(chr(10))[0][:80]}")

        if (idx + 1) % args.save_interval == 0:
            with open(tmp_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            tqdm.write(f"   临时保存: {tmp_path} ({len(results)} 条)")

    pbar.close()

    # ── 保存最终结果 ────────────────────────────────────────────────
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    total    = len(results)
    has_prog = sum(1 for r in results if r['program'] and not r['error'])
    errors   = sum(1 for r in results if r['error'])

    print(f"\n{'='*60}")
    print(f"✅ 生成完成！")
    print(f"   总样本数     : {total}")
    print(f"   program 成功 : {has_prog} ({has_prog/total*100:.1f}%)")
    print(f"   生成错误     : {errors}")
    print(f"   输出文件     : {args.output}")
    print(f"{'='*60}")

    print("\n示例输出（前3条）：")
    for r in results[:3]:
        print(f"\n  scan_id   : {r['scan_id']}")
        print(f"  caption   : {r['caption'][:70]}...")
        print(f"  target_id : {r['target_id']}")
        print(f"  program   :\n    " +
              (r['program'] or r['error'] or '').replace('\n', '\n    '))


if __name__ == '__main__':
    main()