from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np

from .geometry import pixel_scale_m_per_px


@dataclass
class CandidateMetric:
    u: int
    v: int
    yaw: float
    score: float
    width: float
    support_length: float
    flatness: float
    symmetry: float


def _line_extent(mask: np.ndarray, u: int, v: int, angle: float, max_steps: int = 250) -> Tuple[int, int]:
    du = math.cos(angle)
    dv = math.sin(angle)

    def march(direction: float) -> int:
        for step in range(1, max_steps + 1):
            x = int(round(u + direction * du * step))
            y = int(round(v + direction * dv * step))
            if y < 0 or y >= mask.shape[0] or x < 0 or x >= mask.shape[1]:
                return step - 1
            if mask[y, x] == 0:
                return step - 1
        return max_steps

    return march(-1.0), march(1.0)


def _local_maxima(dist_map: np.ndarray, min_clearance_px: float, max_points: int) -> List[Tuple[int, int]]:
    if dist_map.size == 0:
        return []

    dilated = cv2.dilate(dist_map, np.ones((9, 9), dtype=np.uint8))
    maxima_mask = np.logical_and(dist_map >= dilated - 1e-6, dist_map >= min_clearance_px)
    ys, xs = np.nonzero(maxima_mask)
    if len(xs) == 0:
        return []

    order = np.argsort(dist_map[ys, xs])[::-1]
    selected: List[Tuple[int, int]] = []
    for idx in order:
        point = (int(xs[idx]), int(ys[idx]))
        if all((point[0] - other[0]) ** 2 + (point[1] - other[1]) ** 2 >= 64 for other in selected):
            selected.append(point)
        if len(selected) >= max_points:
            break
    return selected


def _normalize_positive(value: float, target: float) -> float:
    if target <= 0.0:
        return 0.0
    return max(0.0, min(1.0, value / target))


def _score_width(width: float, grasp_width_min: float, grasp_width_max: float) -> float:
    optimal = 0.5 * (grasp_width_min + grasp_width_max)
    spread = max(1e-6, 0.5 * (grasp_width_max - grasp_width_min))
    return max(0.0, 1.0 - abs(width - optimal) / spread)


def _flatness_from_depth(depth: np.ndarray, mask: np.ndarray, u: int, v: int, window_radius: int) -> float:
    x0 = max(0, u - window_radius)
    x1 = min(depth.shape[1], u + window_radius + 1)
    y0 = max(0, v - window_radius)
    y1 = min(depth.shape[0], v + window_radius + 1)
    depth_window = depth[y0:y1, x0:x1]
    mask_window = mask[y0:y1, x0:x1] > 0
    samples = depth_window[np.logical_and(mask_window, np.isfinite(depth_window))]
    if samples.size < 6:
        return float("inf")
    return float(np.std(samples))


def generate_candidates(
    mask: np.ndarray,
    contour: np.ndarray,
    depth_image_m: np.ndarray,
    centroid_px: Tuple[int, int],
    base_angle: float,
    intrinsics: Sequence[float],
    angle_delta_deg: float,
    min_clearance_px: float,
    grasp_width_min: float,
    grasp_width_max: float,
    support_length_min: float,
    max_local_depth_std: float,
    score_weights: Sequence[float],
    max_points: int = 8,
) -> List[CandidateMetric]:
    del contour
    distance_mask = (mask > 0).astype(np.uint8)
    dist_map = cv2.distanceTransform(distance_mask, cv2.DIST_L2, 5)
    points = _local_maxima(dist_map, min_clearance_px, max_points)
    centroid_u, centroid_v = int(round(centroid_px[0])), int(round(centroid_px[1]))
    if (
        0 <= centroid_v < mask.shape[0]
        and 0 <= centroid_u < mask.shape[1]
        and mask[centroid_v, centroid_u] > 0
        and dist_map[centroid_v, centroid_u] >= min_clearance_px
        and (centroid_u, centroid_v) not in points
    ):
        points.insert(0, (centroid_u, centroid_v))
    if not points:
        return []

    angle_delta = math.radians(angle_delta_deg)
    candidate_yaws = [base_angle, base_angle + math.pi * 0.5, base_angle - angle_delta, base_angle + angle_delta]
    candidates: List[CandidateMetric] = []
    for u, v in points:
        depth_m = float(depth_image_m[v, u])
        if not np.isfinite(depth_m) or depth_m <= 0.0:
            continue

        meters_per_pixel = pixel_scale_m_per_px(depth_m, intrinsics)
        for yaw in candidate_yaws:
            width_left_px, width_right_px = _line_extent(mask, u, v, yaw + math.pi * 0.5)
            support_left_px, support_right_px = _line_extent(mask, u, v, yaw)

            width = (width_left_px + width_right_px) * meters_per_pixel
            support = (support_left_px + support_right_px) * meters_per_pixel
            if width < grasp_width_min or width > grasp_width_max:
                continue
            if support < support_length_min:
                continue

            flatness = _flatness_from_depth(depth_image_m, mask, u, v, window_radius=6)
            if not np.isfinite(flatness) or flatness > max_local_depth_std:
                continue

            symmetry_gap = abs(width_left_px - width_right_px) * meters_per_pixel
            symmetry = max(0.0, 1.0 - symmetry_gap / max(width, 1e-6))
            clearance_score = _normalize_positive(float(dist_map[v, u]), min_clearance_px * 2.5)
            support_score = _normalize_positive(support, support_length_min * 1.8)
            flatness_score = max(0.0, 1.0 - flatness / max_local_depth_std)
            width_score = _score_width(width, grasp_width_min, grasp_width_max)
            reach_score = 1.0

            w1, w2, w3, w4, w5, w6 = score_weights
            score = (
                w1 * clearance_score
                + w2 * width_score
                + w3 * support_score
                + w4 * flatness_score
                + w5 * symmetry
                + w6 * reach_score
            )

            candidates.append(
                CandidateMetric(
                    u=u,
                    v=v,
                    yaw=yaw,
                    score=float(score),
                    width=float(width),
                    support_length=float(support),
                    flatness=float(flatness),
                    symmetry=float(symmetry),
                )
            )

    candidates.sort(key=lambda candidate: candidate.score, reverse=True)
    return candidates
