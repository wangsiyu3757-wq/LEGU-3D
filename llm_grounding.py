"""
LEGU-3D: NR3D grounding evaluation (OLM + DCM)
==============================================

Given LLM-parsed query programs (QPM output, see gen_visprogapple.py), this
script grounds each Nr3D query with a text-only LLM:
  1. Candidate objects of the target class are listed in a table sorted by
     distance to the nearest anchor (OLM).
  2. Each candidate's distance, W x H x D box size and quadrant direction
     relative to each anchor are written into the prompt as text (DCM).
  3. Bidirectional synonym matching for target and anchor labels.

Object boxes and class labels come from data/scannet/feats_3d.pkl
(GT boxes + GT class labels, produced by the ZSVG3D preprocessing).
Reports Nr3D Acc@25 / easy / hard / view-dependent / view-independent.
"""

import os
import re
import json
import pickle
import logging
import numpy as np
import requests
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm

from zsvg.direction import describe_relation, SCHEMES as DIRECTION_SCHEMES
from zsvg.geometry import calc_iou
from zsvg.run_ckpt import run_with_ckpt, sum_deltas
from zsvg.target_resolve import resolve_target


LLM_URL = "http://127.0.0.1:8000/v1/chat/completions"
LLM_MODEL = "Qwen/Qwen3-32B-AWQ"
FEATS_PKL = "data/scannet/feats_3d.pkl"
# vLLM ignores the key unless the server was started with --api-key.
LLM_HEADERS = {
    "Authorization": f"Bearer {os.environ.get('VLLM_API_KEY', 'EMPTY')}",
    "Content-Type": "application/json",
}

SYSTEM_PROMPT = (
    "/no_think "
    "You are a 3D spatial grounding assistant. "
    "Given a description and candidate objects with 3D positions and sizes, select the best match. "
    "Coordinate system: x=right(+)/left(-), y=forward(+)/backward(-), z=up(+)/down(-). "
    "Output ONLY JSON: "
    '{"target_id": <int>, "confidence": <float>, "reasoning": "<one sentence>"}'
)


SYNONYMS = {
    "sofa": ["couch", "settee"],
    "couch": ["sofa", "settee"],
    "chair": ["seat", "stool"],
    "table": ["desk", "counter"],
    "desk": ["table", "workstation"],
    "cabinet": ["cupboard", "locker", "wardrobe"],
    "wardrobe": ["closet", "cabinet"],
    "shelf": ["shelving", "bookshelf", "bookcase"],
    "bookshelf": ["bookcase", "shelf"],
    "bed": ["mattress"],
    "nightstand": ["bedside table", "side table"],
    "dresser": ["chest of drawers", "drawers"],
    "trash can": ["bin", "garbage bin", "waste bin", "trashcan"],
    "bin": ["trash can", "garbage bin", "waste bin"],
    "tv": ["television", "monitor", "screen"],
    "monitor": ["screen", "display", "tv"],
    "whiteboard": ["board"],
    "refrigerator": ["fridge"],
    "sink": ["basin"],
    "toilet": ["commode"],
    "door": ["doorway"],
    "window": ["glass"],
    "pillow": ["cushion"],
    "lamp": ["light"],
    "picture": ["painting", "photo", "frame"],
    "box": ["container"],
    "curtain": ["drape"],
    "counter": ["countertop", "table"],
    "ottoman": ["footstool"],
}


def _extract_loc_objects(program: str) -> List[str]:
    matches = re.findall(r"LOC\(object='([^']+)'\)", program or "")
    return list(dict.fromkeys(matches))


def _norm_label(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[_-]+", " ", str(text).lower())).strip()


def _same_obj_id(a, b) -> bool:
    return str(a) == str(b)


def _box_minmax_to_center_size(box: np.ndarray) -> np.ndarray:
    """Convert [x1,y1,z1,x2,y2,z2] to [cx,cy,cz,w,h,d]."""
    box = np.array(box, dtype=float)
    bmin = box[:3]
    bmax = box[3:6]
    center = (bmin + bmax) / 2.0
    size = np.maximum(bmax - bmin, 0.0)
    return np.concatenate([center, size])


def _target_box_from_program(program: Dict) -> Optional[np.ndarray]:
    """Read new JSON target_box/gt_box/bbox fields, stored as min-max boxes."""
    for key in ("target_box", "gt_box", "bbox"):
        value = program.get(key)
        if value is not None:
            arr = np.array(value, dtype=float).reshape(-1)
            if len(arr) >= 6:
                return _box_minmax_to_center_size(arr[:6])
    return None


def _get_aliases(label: str) -> set:
    label = _norm_label(label)
    aliases = {label}
    pending = [label]
    while pending:
        cur = pending.pop()
        for alias in SYNONYMS.get(cur, []):
            alias = _norm_label(alias)
            if alias not in aliases:
                aliases.add(alias)
                pending.append(alias)
        for key, values in SYNONYMS.items():
            key = _norm_label(key)
            values = {_norm_label(v) for v in values}
            if cur == key or cur in values:
                if key not in aliases:
                    aliases.add(key)
                    pending.append(key)
                for value in values:
                    if value not in aliases:
                        aliases.add(value)
                        pending.append(value)
    return aliases


def _labels_match(query: str, label: str) -> bool:
    q_aliases = _get_aliases(query)
    l_aliases = _get_aliases(label)
    for q in q_aliases:
        for l in l_aliases:
            if q == l or q in l or l in q:
                return True
    return False


def _call_llm(prompt: str) -> Optional[Dict]:
    try:
        resp = requests.post(
            LLM_URL,
            headers=LLM_HEADERS,
            json={
                "model": LLM_MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.0,
                "max_tokens": 256,
            },
            timeout=120,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        match = re.search(r"\{.*?\"target_id\".*?\}", raw, re.DOTALL)
        if match:
            raw = match.group(0)
        return json.loads(raw)
    except Exception as exc:
        print(f"    [LTS] LLM call failed: {exc}")
        return None


class LTSGrounder:
    _feats = None

    def __init__(
        self,
        top_k: int = 15,
        use_dcr: bool = True,
        direction_scheme: str = "quadrant",
        target_trace: bool = False,
        feats_pkl: str = FEATS_PKL,
    ):
        self.top_k = top_k
        self.use_dcr = use_dcr
        self.direction_scheme = direction_scheme
        self.target_trace = target_trace
        self.feats_pkl = feats_pkl
        self.cache: Dict[Tuple[str, str], Tuple[List[float], int]] = {}

    def _get_feats(self):
        if LTSGrounder._feats is None:
            with open(self.feats_pkl, "rb") as f:
                LTSGrounder._feats = pickle.load(f)
        return LTSGrounder._feats

    def _get_labels(self, scan_id: str) -> List[str]:
        feats = self._get_feats()
        item = feats[scan_id]
        obj_ids = item["obj_ids"]
        for key in ("batch_labels", "labels", "inst_labels", "obj_labels"):
            labels = item.get(key)
            if labels is not None and len(labels) >= len(obj_ids):
                return [_norm_label(labels[i]) or "object" for i in range(len(obj_ids))]
        raise KeyError(
            f"{self.feats_pkl}: scan {scan_id} has no per-object class labels "
            f"(expected one of batch_labels/labels/inst_labels/obj_labels)."
        )

    def _match_anchor(self, atype: str, objects: List[Dict]) -> Optional[Dict]:
        for obj in objects:
            if _labels_match(atype, obj["obj_name"]):
                return obj
        return None

    def _sort_by_dcr(self, candidates: List[Dict], anchors: List[Dict]) -> List[Dict]:
        if not self.use_dcr or not anchors:
            return candidates

        def min_anchor_dist(cand: Dict) -> float:
            c = np.array(cand["obj_loc"][:3], dtype=float)
            return min(float(np.linalg.norm(c - np.array(a["obj_loc"][:3], dtype=float))) for a in anchors)

        return sorted(candidates, key=lambda cand: (min_anchor_dist(cand), cand["obj_id"]))

    def _fallback_first_candidate(self, candidates: List[Dict]) -> Dict:
        return candidates[0]

    def ground(self, scan_id: str, caption: str, program: str):
        self.last_llm_ok = True
        key = (scan_id, caption)
        if key in self.cache:
            return self.cache[key]

        feats = self._get_feats()
        if scan_id not in feats:
            return None, None

        obj_ids = feats[scan_id]["obj_ids"]
        inst_locs = feats[scan_id]["inst_locs"]
        center = np.array(feats[scan_id].get("center", [0.0, 0.0, 0.0])[:3], dtype=float)
        labels = self._get_labels(scan_id)

        all_objs = []
        for i, obj_id in enumerate(obj_ids):
            loc = np.array(inst_locs[i], dtype=float)
            size = loc[3:6] if len(loc) >= 6 else np.array([0.0, 0.0, 0.0])
            all_objs.append({
                "obj_id": obj_id,
                "obj_name": labels[i],
                "obj_loc": loc,
                "obj_size": size,
            })

        loc_objects = _extract_loc_objects(program)
        if self.target_trace:
            target_type, anchor_types = resolve_target(program, loc_objects)
            target_type = target_type.lower()
        else:
            target_type = loc_objects[0].lower() if loc_objects else ""
            anchor_types = loc_objects[1:] if len(loc_objects) > 1 else []

        candidates = [obj for obj in all_objs if target_type and _labels_match(target_type, obj["obj_name"])]
        if not candidates:
            candidates = all_objs[:]

        anchors = [anchor for anchor in (self._match_anchor(t, all_objs) for t in anchor_types) if anchor]

        candidates = self._sort_by_dcr(candidates, anchors)
        candidates = candidates[: self.top_k]

        anchor_lines = []
        for anchor in anchors:
            aloc = np.array(anchor["obj_loc"][:3], dtype=float)
            asize = np.array(anchor["obj_size"], dtype=float)
            anchor_lines.append(
                f"- id={anchor['obj_id']} '{anchor['obj_name']}' "
                f"pos=({aloc[0]:.2f},{aloc[1]:.2f},{aloc[2]:.2f}) "
                f"size=({asize[0]:.2f} x {asize[1]:.2f} x {asize[2]:.2f})m"
            )

        cand_lines = []
        for cand in candidates:
            cloc = np.array(cand["obj_loc"][:3], dtype=float)
            csize = np.array(cand["obj_size"], dtype=float)
            rels = [
                "vs '" + anchor["obj_name"] + "': " +
                describe_relation(
                    cloc, np.array(anchor["obj_loc"][:3], dtype=float), center,
                    scheme=self.direction_scheme,
                )
                for anchor in anchors
            ]
            rel_str = " | ".join(rels) if rels else "no anchor"
            cand_lines.append(
                f"- id={cand['obj_id']} '{cand['obj_name']}' "
                f"pos=({cloc[0]:.2f},{cloc[1]:.2f},{cloc[2]:.2f}) "
                f"size=({csize[0]:.2f} x {csize[1]:.2f} x {csize[2]:.2f})m | {rel_str}"
            )

        order_note = "DCR nearest-anchor order" if anchors and self.use_dcr else "object-ID order"
        prompt = (
            f"Description: {caption}\n\n"
            "Anchor objects:\n" +
            ("\n".join(anchor_lines) if anchor_lines else "(none)") +
            f"\n\nCandidate objects ({order_note}; each has 3D position and size):\n" +
            "\n".join(cand_lines) +
            "\n\nWhich candidate id best matches the description? Output JSON only."
        )

        result = _call_llm(prompt)

        if result is None:
            self.last_llm_ok = False
            chosen = self._fallback_first_candidate(candidates)
            return list(np.array(chosen["obj_loc"], dtype=float)), chosen["obj_id"]

        chosen_id = result.get("target_id")
        for obj in all_objs:
            if _same_obj_id(obj["obj_id"], chosen_id):
                loc = list(np.array(obj["obj_loc"], dtype=float))
                self.cache[key] = (loc, chosen_id)
                return loc, chosen_id

        chosen = self._fallback_first_candidate(candidates)
        return list(np.array(chosen["obj_loc"], dtype=float)), chosen["obj_id"]


def run_lts_eval(
    prog_path: str,
    exp_name: str = "legu3d_nr3d",
    llm_url: str = LLM_URL,
    llm_model: str = LLM_MODEL,
    max_samples: Optional[int] = None,
    top_k: int = 15,
    use_dcr: bool = True,
    direction_scheme: str = "quadrant",
    target_trace: bool = False,
    feats_pkl: str = FEATS_PKL,
):
    global LLM_URL, LLM_MODEL
    LLM_URL = llm_url
    LLM_MODEL = llm_model

    with open(prog_path) as f:
        programs = json.load(f)
    if max_samples:
        programs = programs[:max_samples]

    grounder = LTSGrounder(
        top_k=top_k,
        use_dcr=use_dcr,
        direction_scheme=direction_scheme,
        target_trace=target_trace,
        feats_pkl=feats_pkl,
    )
    feats = grounder._get_feats()

    def eval_one(program):
        d = {}
        scan_id = program["scan_id"]
        target_id = program["target_id"]

        if scan_id not in feats:
            return {"d": {"errors": 1}, "llm_ok": True}

        obj_ids = feats[scan_id]["obj_ids"]
        inst_locs = feats[scan_id]["inst_locs"]

        target_box = _target_box_from_program(program)
        if target_box is None:
            target_idx = next((i for i, obj_id in enumerate(obj_ids) if _same_obj_id(obj_id, target_id)), None)
            if target_idx is None:
                return {"d": {"errors": 1}, "llm_ok": True}
            target_box = inst_locs[target_idx]

        easy = 1 if program.get("easy") else 0
        dep = 1 if program.get("view_dep") else 0
        d["easy_total"], d["dep_total"] = easy, dep

        llm_ok = True
        try:
            pred_box, _ = grounder.ground(
                scan_id,
                program.get("caption", program.get("caption_original", "")),
                program.get("program", ""),
            )
            llm_ok = grounder.last_llm_ok
            if pred_box is None:
                d["errors"] = 1
            else:
                iou = calc_iou(np.array(pred_box), np.array(target_box))
                if iou >= 0.25:
                    d["correct_25"] = d["recall"] = 1
                    d["correct_easy"], d["correct_dep"] = easy, dep
        except Exception as exc:
            print(f"Error {scan_id}: {exc}")
            d["errors"] = 1
        return {"d": d, "llm_ok": llm_ok}

    signature = {
        "script": "llm_grounding.py", "prog_path": prog_path, "max_samples": max_samples,
        "top_k": top_k, "use_dcr": use_dcr, "llm_model": llm_model,
        "direction_scheme": direction_scheme, "target_trace": target_trace,
    }
    records, llm_fallbacks = run_with_ckpt(
        programs, eval_one, f"{exp_name}.ckpt.jsonl", signature, "LEGU-3D NR3D eval")
    c = sum_deltas(records)
    correct_25, correct_easy, correct_dep = c.get("correct_25", 0), c.get("correct_easy", 0), c.get("correct_dep", 0)
    easy_total, dep_total, recall, errors = c.get("easy_total", 0), c.get("dep_total", 0), c.get("recall", 0), c.get("errors", 0)

    total = len(programs)

    logging.basicConfig(
        filename=f"{exp_name}.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(exp_name)
    logger.info("[LEGU-3D] NR3D GT-box evaluation")
    logger.info(f"Easy       {correct_easy / easy_total if easy_total else 0:.4f}  {correct_easy}/{easy_total}")
    logger.info(f"Hard       {(correct_25 - correct_easy) / (total - easy_total) if total > easy_total else 0:.4f}  {correct_25 - correct_easy}/{total - easy_total}")
    logger.info(f"View-Dep   {correct_dep / dep_total if dep_total else 0:.4f}  {correct_dep}/{dep_total}")
    logger.info(f"View-Indep {(correct_25 - correct_dep) / (total - dep_total) if total > dep_total else 0:.4f}  {correct_25 - correct_dep}/{total - dep_total}")
    logger.info(f"Acc@25     {correct_25 / total:.4f}  {correct_25}/{total}")
    logger.info(f"Recall     {recall / total:.4f}  {recall}/{total}")
    logger.info(f"Errors     {errors}")
    logger.info(f"LLM failures scored by fallback: {llm_fallbacks}")

    print("\n[LEGU-3D NR3D]")
    print(f"Acc@25 : {correct_25 / total:.4f} ({correct_25}/{total})")
    print(f"Easy   : {correct_easy / easy_total if easy_total else 0:.4f} ({correct_easy}/{easy_total})")
    print(f"Hard   : {(correct_25 - correct_easy) / (total - easy_total) if total > easy_total else 0:.4f}")
    print(f"View-D : {correct_dep / dep_total if dep_total else 0:.4f} ({correct_dep}/{dep_total})")
    print(f"Recall : {recall / total:.4f} ({recall}/{total})")
    print(f"Errors : {errors}")
    print(f"LLM failures scored by fallback: {llm_fallbacks}")
    print(f"Log    : {exp_name}.log")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LEGU-3D NR3D evaluation (OLM + DCM, text-only LLM)")
    parser.add_argument("--prog_path", default="data/nr3d_qwen3_feats.json",
                        help="QPM programs JSON (output of gen_visprogapple.py).")
    parser.add_argument("--feats_pkl", default=FEATS_PKL,
                        help="ZSVG3D scene features with GT boxes and class labels.")
    parser.add_argument("--exp_name", default="legu3d_nr3d")
    parser.add_argument("--llm_url", default="http://127.0.0.1:8000/v1/chat/completions")
    parser.add_argument("--llm_model", default="Qwen/Qwen3-32B-AWQ")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=15)
    parser.add_argument("--no_dcr", action="store_true",
                        help="Ablation: list candidates in object-ID order instead of distance order.")
    parser.add_argument("--direction-scheme", choices=DIRECTION_SCHEMES, default="quadrant",
                         help="How candidate-vs-anchor direction is phrased in the prompt.")
    parser.add_argument("--target_trace", action="store_true",
                         help="Resolve target_type by tracing the program's TARGET= chain back "
                              "to its LOC() call, instead of assuming the first LOC() mentioned "
                              "is the target. Fixes cases where the caption names a landmark "
                              "before the actual target.")
    args = parser.parse_args()

    run_lts_eval(
        prog_path=args.prog_path,
        exp_name=args.exp_name,
        llm_url=args.llm_url,
        llm_model=args.llm_model,
        max_samples=args.max_samples,
        top_k=args.top_k,
        use_dcr=not args.no_dcr,
        target_trace=args.target_trace,
        direction_scheme=args.direction_scheme,
        feats_pkl=args.feats_pkl,
    )
