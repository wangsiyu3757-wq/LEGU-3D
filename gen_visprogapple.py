import os
import re
import sys
import json
import time
import argparse

import jsonlines
import numpy as np
import requests
from tqdm import tqdm
from typing import List, Dict, Optional


# =====================================================================
# 1. 数据加载（对齐 ZSVG3D）
# =====================================================================

def load_nr3d_val(nr3d_path: str, split_path: str) -> List[Dict]:
    with open(split_path, 'r') as f:
        val_scan_ids = set(line.strip() for line in f if line.strip())

    samples = []
    with jsonlines.open(nr3d_path, 'r') as reader:
        for item in reader:
            if item.get('scan_id') in val_scan_ids:
                samples.append(item)

    print(f"✅ 加载 Nr3D val: {len(samples)} 条样本 | {len(val_scan_ids)} 个场景")
    return samples


def decode_stimulus_string(s: str):
    """解析 stimulus_id → (scene_id, label, n_objects, target_id, distractor_ids)"""
    parts = s.split('-', maxsplit=4)
    if len(parts) == 4:
        scene_id, instance_label, n_objects, target_id = parts
        distractors_ids = []
    else:
        scene_id, instance_label, n_objects, target_id, distractors_ids = parts
        distractors_ids = [int(i) for i in distractors_ids.split('-') if i != '']

    return (
        scene_id,
        instance_label.replace('_', ' '),
        int(n_objects),
        int(target_id),
        distractors_ids
    )


def is_view_dependent(tokens: List[str]) -> bool:
    vd = {'front', 'behind', 'back', 'right', 'left', 'facing',
          'leftmost', 'rightmost', 'looking', 'across'}
    return bool(set(t.lower() for t in tokens) & vd)


# =====================================================================
# 2. 场景数据加载（读取 target_box）
# =====================================================================

def load_target_box(scene_root: str, scan_id: str, target_id: int) -> Optional[List[float]]:
    """
    从场景数据中读取 target_id 对应实例的 bbox。
    返回 [x1, y1, z1, x2, y2, z2] 或 None。
    """
    # 尝试 npz 格式
    npz_path = os.path.join(scene_root, f"{scan_id}.npz")
    if os.path.exists(npz_path):
        try:
            data = np.load(npz_path, allow_pickle=True)
            pcds = data['ins_pcds']
            if target_id < len(pcds) and pcds[target_id] is not None:
                xyz = np.array(pcds[target_id])[:, :3].astype(np.float32)
                if len(xyz) >= 3:
                    bmin = xyz.min(axis=0)
                    bmax = xyz.max(axis=0)
                    return [round(float(v), 4) for v in list(bmin) + list(bmax)]
        except Exception:
            pass

    # 尝试目录格式
    for d in [os.path.join(scene_root, scan_id, scan_id),
              os.path.join(scene_root, scan_id)]:
        pcd_path = os.path.join(d, 'ins_pcds.npy')
        if os.path.exists(pcd_path):
            try:
                pcds = np.load(pcd_path, allow_pickle=True)
                if target_id < len(pcds) and pcds[target_id] is not None:
                    xyz = np.array(pcds[target_id])[:, :3].astype(np.float32)
                    if len(xyz) >= 3:
                        bmin = xyz.min(axis=0)
                        bmax = xyz.max(axis=0)
                        return [round(float(v), 4) for v in list(bmin) + list(bmax)]
            except Exception:
                pass

    return None



# =====================================================================
# 3. Metric Caption 增强（把模糊空间词替换为真实3D距离）
# =====================================================================

# 常见参考物体关键词（与 eval 脚本保持一致）
REF_KEYWORDS = [
    'toilet', 'desk', 'chair', 'table', 'sofa', 'couch',
    'window', 'wall', 'door', 'bed', 'sink', 'cabinet',
    'shelf', 'monitor', 'tv', 'piano', 'refrigerator',
    'whiteboard', 'counter', 'dresser', 'nightstand', 'pillow',
]

# 空间词 pattern — 长短语优先，避免部分匹配
SPATIAL_PATTERNS = [
    # left/right 从长到短，避免 "on the right hand side" 被截断
    (r'\bto the far left\b',   'to the far left'),
    (r'\bto the left\b',       'to the left'),
    (r'\bon the far left\b',   'on the far left'),
    (r'\bon the left\b',       'on the left'),
    (r'\bto the far right\b',  'to the far right'),
    (r'\bto the right\b',      'to the right'),
    (r'\bon the far right\b',  'on the far right'),
    (r'\bon the right\b',      'on the right'),
    # "on the far end/side" 包含介词整体匹配
    (r'\bon the far end\b',    'on the far end'),
    (r'\bon the far side\b',   'on the far side'),
    # leftmost / rightmost
    (r'\bleftmost\b',   'leftmost'),
    (r'\brightmost\b',  'rightmost'),
    # in front of / facing
    (r'\bin front of\b', 'in front of'),
    (r'\bfacing\b',      'facing'),
    # behind / at the back / at the rear
    (r'\bbehind\b',         'behind'),
    (r'\bat the back of\b', 'at the back of'),
    (r'\bat the back\b',    'at the back'),
    (r'\bat the rear and\b','at the rear and'),
    (r'\bat the rear\b',    'at the rear'),
    # next to / beside / close to / near
    (r'\bnext to\b',   'next to'),
    (r'\bbeside\b',    'beside'),
    (r'\bclose to\b',  'close to'),
    (r'\bnear\b',      'near'),
    # standalone farthest / far end / far side
    (r'\bfarthest\b',  'farthest'),
    (r'\bfar end\b',   'far end'),
    (r'\bfar side\b',  'far side'),
]



def load_scene_objects(scene_root: str, scan_id: str) -> Optional[Dict]:
    """
    加载场景中所有实例的中心点坐标和标签。
    返回 {instance_id: {'label': str, 'center': np.ndarray, 'bbox_min': ..., 'bbox_max': ...}}
    """
    objects = {}

    # npz 格式
    npz_path = os.path.join(scene_root, f"{scan_id}.npz")
    if os.path.exists(npz_path):
        try:
            data   = np.load(npz_path, allow_pickle=True)
            pcds   = data['ins_pcds']
            labels = data['ins_labels']
            for i, (pcd, label) in enumerate(zip(pcds, labels)):
                if pcd is None or len(pcd) < 5:
                    continue
                xyz = np.array(pcd)[:, :3].astype(np.float32)
                objects[i] = {
                    'label':    str(label).lower().strip(),
                    'center':   xyz.mean(axis=0),
                    'bbox_min': xyz.min(axis=0),
                    'bbox_max': xyz.max(axis=0),
                }
            return objects if objects else None
        except Exception:
            pass

    # 目录格式
    for d in [os.path.join(scene_root, scan_id, scan_id),
              os.path.join(scene_root, scan_id)]:
        pcd_path   = os.path.join(d, 'ins_pcds.npy')
        label_path = os.path.join(d, 'ins_labels.npy')
        if os.path.exists(pcd_path) and os.path.exists(label_path):
            try:
                pcds   = np.load(pcd_path,   allow_pickle=True)
                labels = np.load(label_path, allow_pickle=True)
                for i, (pcd, label) in enumerate(zip(pcds, labels)):
                    if pcd is None or len(pcd) < 5:
                        continue
                    xyz = np.array(pcd)[:, :3].astype(np.float32)
                    objects[i] = {
                        'label':    str(label).lower().strip(),
                        'center':   xyz.mean(axis=0),
                        'bbox_min': xyz.min(axis=0),
                        'bbox_max': xyz.max(axis=0),
                    }
                return objects if objects else None
            except Exception:
                pass

    return None


def find_ref_object(objects: Dict, caption_lower: str, exclude_id: int) -> Optional[Dict]:
    """
    在 caption 中找到最合适的参考物体实例。
    优先级：caption 中出现 + 实例标签匹配。
    """
    best = None
    best_priority = -1

    for priority, kw in enumerate(reversed(REF_KEYWORDS)):  # 越靠前优先级越高
        if kw not in caption_lower:
            continue
        # 找场景中所有匹配该关键词的实例
        matches = [
            obj for iid, obj in objects.items()
            if kw in obj['label'] and iid != exclude_id
        ]
        if matches and priority > best_priority:
            # 取最近的一个（如果有多个同类型）
            best = matches[0]
            best_priority = priority

    return best


def compute_metric_distance(target_center: np.ndarray, ref_center: np.ndarray) -> float:
    """计算两个物体中心的欧氏距离（米）"""
    return float(np.linalg.norm(target_center - ref_center))


def enhance_caption_with_metrics(
    caption: str,
    target_id: int,
    scene_objects: Optional[Dict],
) -> str:
    """
    把 caption 中的模糊空间词替换为真实3D距离。

    例如:
      "the chair to the far left of the desk"
      → "the chair 2.3m to the left of the desk"

      "if looking from the toilet, the towel on the left"
      → "if looking from the toilet, the towel 1.2m to the left"
    """
    if scene_objects is None:
        return caption  # 没有场景数据，直接返回原始 caption

    target_obj = scene_objects.get(target_id)
    if target_obj is None:
        return caption

    caption_lower = caption.lower()

    # 找参考物体
    ref_obj = find_ref_object(scene_objects, caption_lower, exclude_id=target_id)
    if ref_obj is None:
        return caption  # 没有参考物体，无法计算距离

    # 计算真实距离
    dist = compute_metric_distance(target_obj['center'], ref_obj['center'])
    dist_str = f"{dist:.1f}m"

    # 已有数字距离 → 不再插入
    if re.search(r'[0-9]+[.]?[0-9]* *(meter|meters|cm|feet|ft)', caption_lower):
        return caption

    # 收集所有匹配位置
    result = caption
    all_matches = []
    for pattern, _ in SPATIAL_PATTERNS:
        for m in re.finditer(pattern, result, re.IGNORECASE):
            all_matches.append((m.start(), m.end(), m.group()))

    if not all_matches:
        return caption

    # 选第一个不在 "of X" 嵌入从句里的匹配
    best_pos = None
    for start_pos, end_pos, matched in sorted(all_matches):
        prefix = result[:start_pos].strip()
        if re.search(r'\bof\b\s*\w+\s*$', prefix, re.IGNORECASE):
            continue
        best_pos = start_pos
        break

    if best_pos is None:
        best_pos = sorted(all_matches)[0][0]

    return result[:best_pos] + dist_str + " " + result[best_pos:]


def load_scene_objects_from_feats(feats_data: dict, scan_id: str) -> Optional[Dict]:
    """从 feats_3d.pkl 加载场景物体（与评估数据坐标系一致）"""
    if feats_data is None or scan_id not in feats_data:
        return None
    obj_ids   = feats_data[scan_id]['obj_ids']
    inst_locs = feats_data[scan_id]['inst_locs']  # [cx,cy,cz,w,h,d]
    # batch_labels 可能包含类别信息
    batch_labels = feats_data[scan_id].get('batch_labels', [])

    objects = {}
    for i, oid in enumerate(obj_ids):
        loc = inst_locs[i]
        cx, cy, cz = float(loc[0]), float(loc[1]), float(loc[2])
        w,  h,  d  = float(loc[3]), float(loc[4]), float(loc[5])
        center = np.array([cx, cy, cz])
        bmin   = np.array([cx-w/2, cy-h/2, cz-d/2])
        bmax   = np.array([cx+w/2, cy+h/2, cz+d/2])
        label  = str(batch_labels[i]).lower() if i < len(batch_labels) else 'object'
        objects[oid] = {
            'label':    label,
            'center':   center,
            'bbox_min': bmin,
            'bbox_max': bmax,
        }
    return objects if objects else None

# =====================================================================
# 4. Program 生成（调用本地 vLLM Completions API）
# =====================================================================

def build_prompt(system_prompt: str, caption: str) -> str:
    """构建 Completions API prompt，以 BOX0=LOC 结尾强制续写"""
    return (
        f"{system_prompt}\n"
        f"Description: {caption}\n"
        "Program:\n"
        "BOX0=LOC"
    )


def postprocess_program(raw: str) -> str:
    """
    后处理生成的 program：
    - 修复无引号的 LOC(red cup) → LOC(object='red cup')
    - 只保留有效代码行（BOX* / TARGET= / CENTER=）
    - 找到 TARGET= 后停止，不要提前截断
    """
    full = "BOX0=LOC" + raw

    lines = []
    for line in full.split('\n'):
        line = line.strip()
        if not line:
            continue  # 跳过空行，不截断

        # 跳过注释行
        if line.startswith('#'):
            continue

        # 修复无 object= 的 LOC
        if 'LOC(' in line and 'object=' not in line:
            inner = line.split('LOC(', 1)[1].rstrip(')').strip().strip("'\"")
            inner = inner.replace('_', ' ').lower()
            prefix = line.split('LOC')[0]
            line = f"{prefix}LOC(object='{inner}')"

        # 只保留有效代码行
        if re.match(r'^(BOX[0-9]+|TARGET|CENTER)[ ]*=', line):
            lines.append(line)
            if line.startswith('TARGET='):
                break  # TARGET= 是终止符，找到后停止
        else:
            # 非代码行：只有在已有 BOX 行之后，且连续出现才截断
            # 单行非代码不截断（可能是 LLM 的中间思考）
            continue

    return '\n'.join(lines)


def generate_program(
    caption: str,
    system_prompt: str,
    llm_url: str,
    llm_model: str,
    max_retries: int = 3,
) -> str:
    """调用 vLLM Completions API 生成 visual program."""
    url = f"{llm_url.rstrip('/')}/completions"
    headers = {
        "Authorization": f"Bearer {os.environ.get('VLLM_API_KEY', 'EMPTY')}",
        "Content-Type": "application/json"
    }
    prompt = build_prompt(system_prompt, caption)

    for attempt in range(max_retries):
        try:
            payload = {
                "model": llm_model,
                "prompt": prompt,
                "temperature": 0.0,
                "max_tokens": 256,  # was 1028; the longest training example is ~90 tokens,
                                    # 256 is generous headroom without blowing the context budget
                "stop": ["\n\n", "Description:", "# EXAMPLE", "# TASK"],
                "echo": False,
            }
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["text"].strip()
            program = postprocess_program(raw)

            # 验证最低有效格式
            if "BOX0=LOC" in program and "TARGET=" in program:
                return program

            # 格式不对但已达最大重试 → 返回原始结果
            if attempt == max_retries - 1:
                if program:
                    return program
                return f"ERROR: empty program. raw={raw[:200]}"

        except Exception as e:
            if attempt == max_retries - 1:
                return f"ERROR: {str(e)}"

        time.sleep(1)

    return "ERROR: max retries exceeded"


# =====================================================================
# 5. 主流程
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Nr3D Visual Program 生成器")
    parser.add_argument('--nr3d_path',   default='data/nr3d.jsonl')
    parser.add_argument('--split_path',  default='data/scannet/splits/scannetv2_val.txt')
    parser.add_argument('--scene_root',  default='data/scannet',
                        help='场景数据根目录（用于读取 target_box）')
    parser.add_argument('--use_feats',   action='store_true',
                        help='用 feats_3d.pkl 计算距离（与评估数据一致）')
    parser.add_argument('--prompt_path', default='prompts/prompt_n32.txt',
                        help='System prompt 文件路径')
    parser.add_argument('--output',      default='data/nr3d_programs.json',
                        help='输出 JSON 文件路径')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='最大处理样本数（默认全量）')
    parser.add_argument('--save_interval', type=int, default=50,
                        help='每 N 条保存一次临时结果')
    parser.add_argument('--llm_url',    default='http://127.0.0.1:8000/v1')
    parser.add_argument('--llm_model',  default='Qwen/Qwen3-32B-AWQ')
    parser.add_argument('--resume',     action='store_true',
                        help='从已有输出文件断点续跑')
    args = parser.parse_args()

    # ── 加载 system prompt ────────────────────────────────────────────
    if not os.path.exists(args.prompt_path):
        print(f"❌ prompt 文件不存在: {args.prompt_path}")
        sys.exit(1)
    with open(args.prompt_path, 'r') as f:
        system_prompt = f.read().strip()
    print(f"✅ 加载 prompt: {args.prompt_path} ({len(system_prompt)} chars)")

    # ── 加载数据 ──────────────────────────────────────────────────────
    samples = load_nr3d_val(args.nr3d_path, args.split_path)
    if args.max_samples:
        samples = samples[:args.max_samples]
    print(f" 处理样本数: {len(samples)}")

    # ── 断点续跑 ──────────────────────────────────────────────────────
    results = []
    done_ids = set()
    if args.resume and os.path.exists(args.output):
        with open(args.output, 'r') as f:
            results = json.load(f)
        done_ids = {r['target_id'] + '_' + r['scan_id'] for r in results}
        print(f"♻️  续跑: 已完成 {len(done_ids)} 条")

    # ── 生成循环 ──────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    tmp_path = args.output.replace('.json', '_temp.json')

    # 如果使用 feats_3d.pkl
    feats_data = None
    if args.use_feats:
        import pickle
        feats_pkl = os.path.join(args.scene_root, 'feats_3d.pkl')
        if os.path.exists(feats_pkl):
            with open(feats_pkl, 'rb') as f:
                feats_data = pickle.load(f)
            print(f"✅ 加载 feats_3d.pkl: {len(feats_data)} 个场景")
        else:
            print(f"⚠ feats_3d.pkl 不存在: {feats_pkl}, 使用原始场景数据")

    pbar = tqdm(total=len(samples), desc="生成 programs")

    for idx, sample in enumerate(samples):
        pbar.update(1)

        scan_id    = sample['scan_id']
        target_id  = int(sample['target_id'])
        tokens     = sample.get('tokens', [])
        caption    = ' '.join(tokens) if tokens else sample.get('utterance', '')
        item_id    = sample.get('item_id', sample.get('stimulus_id', ''))

        # 跳过已完成
        dedup_key = item_id if item_id else f"{target_id}_{scan_id}_{idx}"
        if dedup_key in done_ids:
            continue

        # 解析难度
        try:
            _, _, n_objects, _, _ = decode_stimulus_string(sample['stimulus_id'])
            easy = (n_objects <= 2)
        except Exception:
            easy = False

        view_dep = is_view_dependent(tokens)

        # 加载场景中所有实例（用于 metric caption 增强 + target_box）
        if args.use_feats:
            scene_objects = load_scene_objects_from_feats(feats_data, scan_id)
        else:
            scene_objects = load_scene_objects(args.scene_root, scan_id)

        # 读取 target_box
        target_box = None
        if scene_objects and target_id in scene_objects:
            obj = scene_objects[target_id]
            bmin, bmax = obj['bbox_min'], obj['bbox_max']
            target_box = [round(float(v), 4) for v in list(bmin) + list(bmax)]
        else:
            target_box = load_target_box(args.scene_root, scan_id, target_id)

        # 增强 caption：把模糊空间词替换为真实3D距离
        metric_caption = enhance_caption_with_metrics(caption, target_id, scene_objects)
        if metric_caption != caption:
            tqdm.write(f"    caption 增强: {caption[:50]}...")
            tqdm.write(f"            → {metric_caption[:60]}...")

        # 生成 program（使用原始 caption，避免距离数字干扰 LLM）
        program = generate_program(
            caption=caption,
            system_prompt=system_prompt,
            llm_url=args.llm_url,
            llm_model=args.llm_model,
        )

        is_error = program.startswith("ERROR")

        result = {
            # ── ZSVG3D 对齐字段 ──────────────────────────────────────
            "scan_id":    scan_id,
            "caption":    metric_caption,   # 增强后的 metric-aware caption
            "caption_original": caption,    # 保留原始 caption 便于对比
            "target_id":  str(target_id),
            "target_box": target_box,       # 从场景数据读取
            "pred_box":   None,             # 待评估阶段填充
            "pred_id":    None,             # 待评估阶段填充
            "pred_name":  None,             # 待评估阶段填充
            "iou":        0.0,              # 待评估阶段填充
            "success":    None,             # 待评估阶段填充
            "easy":       easy,
            "view_dep":   view_dep,
            "program":    program if not is_error else None,
            "error":      program if is_error else None,
            # ── 内部字段 ─────────────────────────────────────────────
            "item_id":    item_id,
        }

        results.append(result)
        done_ids.add(dedup_key)  # keyed by item_id

        # 进度显示
        status = "✅" if not is_error else "❌"
        tqdm.write(f"{status} [{idx+1}/{len(samples)}] {scan_id} | tid={target_id} | "
                   f"{'VD' if view_dep else 'VI'} | {'easy' if easy else 'hard'}")
        if not is_error:
            first_line = program.split('\n')[0]
            tqdm.write(f"   {first_line[:80]}")

        # 定期保存临时结果
        if (idx + 1) % args.save_interval == 0:
            with open(tmp_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            tqdm.write(f"   临时保存: {tmp_path} ({len(results)} 条)")

    pbar.close()

    # ── 保存最终结果 ──────────────────────────────────────────────────
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 统计
    total     = len(results)
    has_prog  = sum(1 for r in results if r['program'] and not r['error'])
    has_box   = sum(1 for r in results if r['target_box'] is not None)
    errors    = sum(1 for r in results if r['error'])
    vd_count  = sum(1 for r in results if r['view_dep'])
    easy_count = sum(1 for r in results if r['easy'])

    print(f"\n{'='*60}")
    print(f"✅ 生成完成！")
    print(f"   总样本数       : {total}")
    print(f"   program 成功   : {has_prog} ({has_prog/total*100:.1f}%)")
    print(f"   target_box 有效: {has_box} ({has_box/total*100:.1f}%)")
    print(f"   生成错误       : {errors}")
    print(f"   视角依赖 (VD)  : {vd_count} ({vd_count/total*100:.1f}%)")
    print(f"   Easy 样本      : {easy_count} ({easy_count/total*100:.1f}%)")
    print(f"   输出文件       : {args.output}")
    print(f"{'='*60}")

    # 打印前3条示例
    print("\n示例输出（前3条）：")
    for r in results[:3]:
        print(f"\n  scan_id   : {r['scan_id']}")
        print(f"  caption   : {r['caption'][:70]}...")
        print(f"  target_id : {r['target_id']}")
        print(f"  target_box: {r['target_box']}")
        print(f"  program   :\n    " +
              (r['program'] or r['error'] or '').replace('\n', '\n    '))


if __name__ == '__main__':
    main()