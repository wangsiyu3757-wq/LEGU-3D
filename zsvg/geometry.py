"""Axis-aligned 3D box utilities."""

import numpy as np


def calc_iou(box_a, box_b) -> float:
    """IoU of two axis-aligned 3D boxes given as [cx, cy, cz, w, h, d].

    Returns 0.0 unless the boxes overlap with positive extent on every axis.
    """
    box_a = np.asarray(box_a, dtype=float)
    box_b = np.asarray(box_b, dtype=float)
    half_a, half_b = box_a[3:6] / 2, box_b[3:6] / 2
    lo = np.maximum(box_a[0:3] - half_a, box_b[0:3] - half_b)
    hi = np.minimum(box_a[0:3] + half_a, box_b[0:3] + half_b)
    if not (hi > lo).all():
        return 0.0
    inter = float(np.prod(hi - lo))
    union = float(np.prod(box_a[3:6])) + float(np.prod(box_b[3:6])) - inter
    return inter / union
