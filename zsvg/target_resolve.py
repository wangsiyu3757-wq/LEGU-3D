"""Resolve the true target category from a visprog-style program.

The existing grounding code takes target_type = the first LOC(object=...) call
that appears in the program text. That breaks whenever the caption names a
landmark before the actual target (a common NR3D pattern: "the bed in front of
you... to your left, the desk... on it, the books you're looking for") --
BOX0 ends up being the landmark, not the target, even though the program's
own TARGET= line correctly points at the real target through its targets=
chain.

resolve_target(program) walks that chain instead: starting from TARGET=...,
follow targets= (or a bare "TARGET=BOXn") through however many relation
functions until it reaches a LOC(object='...') call, and returns that
object's string. Falls back to the old first-LOC behavior (and to the empty
target) whenever the program doesn't parse cleanly, so it can never behave
worse than today's code, only match it or improve on it.
"""

import re
from typing import Dict, List, Optional, Tuple

_ASSIGN_RE = re.compile(r"^\s*(\w+)\s*=\s*(.+?)\s*$")
_LOC_RE = re.compile(r"^LOC\(object=['\"]([^'\"]+)['\"]\)$")
_TARGETS_ARG_RE = re.compile(r"targets=(\w+)")
_BARE_BOX_RE = re.compile(r"^\w+$")

MAX_DEPTH = 20


def _parse_assignments(program: str) -> Dict[str, str]:
    boxes = {}
    for line in program.splitlines():
        m = _ASSIGN_RE.match(line)
        if m:
            boxes[m.group(1)] = m.group(2)
    return boxes


def _resolve_box(name: str, boxes: Dict[str, str], visited: set) -> Optional[str]:
    if name in visited or name not in boxes:
        return None
    visited.add(name)
    if len(visited) > MAX_DEPTH:
        return None

    rhs = boxes[name]
    m = _LOC_RE.match(rhs)
    if m:
        return m.group(1)

    if _BARE_BOX_RE.match(rhs) and rhs in boxes:
        return _resolve_box(rhs, boxes, visited)

    m = _TARGETS_ARG_RE.search(rhs)
    if m:
        return _resolve_box(m.group(1), boxes, visited)

    return None


def resolve_target(program: str, loc_objects: List[str]) -> Tuple[str, List[str]]:
    """Returns (target_type, anchor_types), both as they appear in the program
    (i.e. not yet lowercased). loc_objects is the de-duplicated, in-order list
    of every LOC(object=...) string in the program (what the old code used
    directly). Falls back to loc_objects[0]/loc_objects[1:] if the TARGET=
    chain can't be resolved.
    """
    if not loc_objects:
        return "", []

    boxes = _parse_assignments(program)
    target_rhs = boxes.get("TARGET")
    resolved = None
    if target_rhs is not None:
        if _BARE_BOX_RE.match(target_rhs) and target_rhs in boxes:
            resolved = _resolve_box(target_rhs, boxes, set())
        else:
            m = _TARGETS_ARG_RE.search(target_rhs)
            if m:
                resolved = _resolve_box(m.group(1), boxes, set())

    if resolved is None:
        return loc_objects[0], loc_objects[1:]

    anchors = [o for o in loc_objects if o != resolved]
    return resolved, anchors
