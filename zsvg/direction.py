"""Shared candidate-vs-anchor direction labeling.

Both the NR3D (llm_grounding.py) and ScanRefer (llm_grounding_scanrefer_test.py)
grounders build a per-candidate relation string that gets inserted into the LLM
prompt (the dir_k term in the relation description). This module holds that
logic in one place so the two grounders can't drift apart, and so a new
direction scheme can be added without re-deriving dx/dz per script.

Axes are anchor-relative, not a fixed scene-frame or viewer-relative axis:
  - dz is the candidate's radial offset from the anchor, measured along the line
    from the scene (room) center to the anchor. dz > 0 means the candidate is
    further from the room center than the anchor; dz < 0 means closer.
  - dx is the offset perpendicular to that line (right of the radial direction
    when facing from the room center toward the anchor).

All schemes share the same (dx, dz, h, dist) computed by relative_deltas():
  - "quadrant" (default, original behavior): left/right + further/closer +
    above/below, each reported with its magnitude in meters.
  - "compass": dx/dz collapsed into a single 8-way bearing (N/NE/E/...), where
    N = further from room center, S = closer, E = right, W = left.
  - "compass_cutoff": same 8-way bearing as "compass", but omits the direction
    (like "quadrant" does; "same pos" appears only if the height part is also
    absent) when the horizontal offset sqrt(dx^2 + dz^2) is under DEAD_ZONE_M.
"""

import math

import numpy as np

COMPASS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

SCHEMES = ("quadrant", "compass", "compass_cutoff")

DEAD_ZONE_M = 0.2


def anchor_frame(anchor_loc: np.ndarray, scene_center: np.ndarray):
    """Unit vectors (right, radial) that a candidate's offset from the anchor is projected onto.

    radial points from the scene center to the anchor. If the anchor sits at the
    scene center the direction is undefined and it falls back to the global +y axis.
    """
    radial = anchor_loc - scene_center
    rn_radial = np.linalg.norm(radial)
    radial = radial / rn_radial if rn_radial > 1e-6 else np.array([0.0, 1.0, 0.0])
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(radial, up)
    rn = np.linalg.norm(right)
    right = right / rn if rn > 1e-6 else np.array([1.0, 0.0, 0.0])
    return right, radial


def relative_deltas(target_loc: np.ndarray, anchor_loc: np.ndarray, scene_center: np.ndarray):
    """dx (perpendicular to the radial line), dz (radial offset), h (up), and 3D dist.

    dz > 0: candidate is further from the scene center than the anchor;
    dz < 0: closer.
    """
    diff = target_loc - anchor_loc
    dist = float(np.linalg.norm(diff))
    right, radial = anchor_frame(anchor_loc, scene_center)
    dx = float(np.dot(diff, right))
    dz = float(np.dot(diff, radial))
    h = float(diff[2])
    return dx, dz, h, dist


def compass_label(dx: float, dz: float) -> str:
    # bearing 0 = N = radially further from room center, 90 = E = right, 180 = S = closer, 270 = W = left
    bearing = math.degrees(math.atan2(dx, dz)) % 360.0
    idx = int((bearing / 45.0) + 0.5) % 8
    return COMPASS[idx]


def describe_relation(
    target_loc: np.ndarray,
    anchor_loc: np.ndarray,
    scene_center: np.ndarray,
    scheme: str = "quadrant",
) -> str:
    if scheme not in SCHEMES:
        raise ValueError(f"Unknown direction scheme {scheme!r}, expected one of {SCHEMES}")

    dx, dz, h, dist = relative_deltas(
        np.asarray(target_loc, dtype=float),
        np.asarray(anchor_loc, dtype=float),
        np.asarray(scene_center, dtype=float),
    )

    parts = []
    if scheme == "quadrant":
        if abs(dx) > 0.2:
            parts.append(("right" if dx > 0 else "left") + f" {abs(dx):.1f}m")
        if abs(dz) > 0.2:
            parts.append(("further" if dz > 0 else "closer") + f" {abs(dz):.1f}m")
    elif scheme == "compass":
        parts.append(compass_label(dx, dz))
    else:  # compass_cutoff; inside the dead zone the direction is omitted, like quadrant
        if math.hypot(dx, dz) >= DEAD_ZONE_M:
            parts.append(compass_label(dx, dz))

    if abs(h) > 0.2:
        parts.append(("above" if h > 0 else "below") + f" {abs(h):.1f}m")

    return f"{dist:.1f}m (" + (", ".join(parts) if parts else "same pos") + ")"
