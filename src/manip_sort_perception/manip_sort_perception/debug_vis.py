from __future__ import annotations

import math
from typing import Iterable, Tuple

import cv2
import numpy as np


def draw_roi(image: np.ndarray, roi: Tuple[int, int, int, int]) -> np.ndarray:
    output = image.copy()
    x_min, y_min, x_max, y_max = roi
    cv2.rectangle(output, (x_min, y_min), (x_max, y_max), (255, 255, 255), 2)
    return output


def draw_components(image: np.ndarray, components: Iterable) -> np.ndarray:
    output = image.copy()
    for component in components:
        x, y, w, h = component.bbox
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 255), 1)
        cv2.circle(output, component.centroid_px, 4, (255, 255, 255), -1)
        cv2.putText(
            output,
            component.color_name,
            (x, max(15, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return output


def draw_candidates(image: np.ndarray, candidates: Iterable, best_candidate=None) -> np.ndarray:
    output = image.copy()
    for candidate in candidates:
        radius = 5 if best_candidate is not None and candidate is best_candidate else 3
        color = (0, 255, 0) if best_candidate is not None and candidate is best_candidate else (255, 180, 0)
        center = (candidate.u, candidate.v)
        cv2.circle(output, center, radius, color, -1)
        dx = int(round(20 * math.cos(candidate.yaw)))
        dy = int(round(20 * math.sin(candidate.yaw)))
        cv2.line(output, (center[0] - dx, center[1] - dy), (center[0] + dx, center[1] + dy), color, 2)
    return output
