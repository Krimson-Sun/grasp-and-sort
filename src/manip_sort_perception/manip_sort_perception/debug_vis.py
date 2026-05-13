from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class DecisionOverlayObject:
    object_id: str
    label: str
    contour: np.ndarray
    bbox: Tuple[int, int, int, int]
    centroid_px: Tuple[int, int]
    candidates: Sequence[object]
    best_score: float


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


def _put_text(
    image: np.ndarray,
    text: str,
    origin: Tuple[int, int],
    color: Tuple[int, int, int],
    scale: float = 0.62,
) -> None:
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_decision_overlay(
    image: np.ndarray,
    roi: Tuple[int, int, int, int],
    objects: Sequence[DecisionOverlayObject],
    selected_object_id: str,
    frame_label: str | None = None,
) -> np.ndarray:
    output = image.copy()
    x_min, y_min, x_max, y_max = roi

    shaded = output.copy()
    shaded[:y_min, :] = (shaded[:y_min, :] * 0.45).astype(np.uint8)
    shaded[y_max:, :] = (shaded[y_max:, :] * 0.45).astype(np.uint8)
    shaded[y_min:y_max, :x_min] = (shaded[y_min:y_max, :x_min] * 0.45).astype(np.uint8)
    shaded[y_min:y_max, x_max:] = (shaded[y_min:y_max, x_max:] * 0.45).astype(np.uint8)
    output = shaded
    cv2.rectangle(output, (x_min, y_min), (x_max, y_max), (220, 220, 220), 2)

    selected = next((item for item in objects if item.object_id == selected_object_id), None)
    if selected is None:
        return output

    highlight = output.copy()
    selected_mask = np.zeros(output.shape[:2], dtype=np.uint8)
    cv2.drawContours(selected_mask, [selected.contour], -1, 255, -1)
    highlight[selected_mask > 0] = (155, 255, 190)
    output = cv2.addWeighted(output, 0.76, highlight, 0.24, 0.0)

    for item in objects:
        x, y, w, h = item.bbox
        color = (225, 225, 225)
        thickness = 1
        if item.object_id == selected_object_id:
            color = (30, 240, 120)
            thickness = 3
        cv2.drawContours(output, [item.contour], -1, color, thickness)
        cv2.rectangle(output, (x, y), (x + w, y + h), color, 1)
        label = item.label
        if item.object_id == selected_object_id:
            label += " SELECTED"
        _put_text(output, label, (x, max(28, y - 10)), color, scale=0.58)

    best_candidate = selected.candidates[0] if selected.candidates else None
    output = draw_candidates(output, selected.candidates, best_candidate=best_candidate)
    cv2.circle(output, selected.centroid_px, 6, (255, 255, 255), -1)
    cv2.circle(output, selected.centroid_px, 12, (30, 240, 120), 2)

    panel_bottom = 184 if frame_label else 156
    panel = output.copy()
    cv2.rectangle(panel, (34, 24), (500, panel_bottom), (15, 18, 26), -1)
    output = cv2.addWeighted(output, 0.82, panel, 0.18, 0.0)
    cv2.rectangle(output, (34, 24), (500, panel_bottom), (220, 220, 220), 1)
    if frame_label:
        _put_text(output, frame_label, (54, 60), (255, 255, 255))
    _put_text(
        output,
        f"Selected object: {selected.label}",
        (54, 95 if frame_label else 60),
        (30, 240, 120),
    )
    _put_text(
        output,
        f"Candidates on object: {len(selected.candidates)}",
        (54, 130 if frame_label else 95),
        (255, 255, 255),
    )
    _put_text(
        output,
        f"Best score: {selected.best_score:.3f}",
        (54, 165 if frame_label else 130),
        (255, 255, 255),
    )
    return output
