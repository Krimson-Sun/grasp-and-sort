from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np


@dataclass
class SegmentedComponent:
    color_name: str
    mask: np.ndarray
    contour: np.ndarray
    area_px: int
    centroid_px: Tuple[int, int]
    bbox: Tuple[int, int, int, int]


def _mask_from_ranges(hsv: np.ndarray, hsv_ranges: Iterable[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    combined = None
    for lower, upper in hsv_ranges:
        partial = cv2.inRange(hsv, lower, upper)
        combined = partial if combined is None else cv2.bitwise_or(combined, partial)
    return combined if combined is not None else np.zeros(hsv.shape[:2], dtype=np.uint8)


def _kernel(size: int) -> np.ndarray:
    side = max(1, int(size))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (side, side))


def segment_components(
    rgb_bgr: np.ndarray,
    hsv_thresholds: Dict[str, List[Tuple[np.ndarray, np.ndarray]]],
    morph_open_kernel: int,
    morph_close_kernel: int,
    min_area: int,
) -> List[SegmentedComponent]:
    hsv = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2HSV)
    open_kernel = _kernel(morph_open_kernel)
    close_kernel = _kernel(morph_close_kernel)

    components: List[SegmentedComponent] = []
    for color_name, ranges in hsv_thresholds.items():
        mask = _mask_from_ranges(hsv, ranges)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for label in range(1, num_labels):
            area_px = int(stats[label, cv2.CC_STAT_AREA])
            if area_px < min_area:
                continue

            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            w = int(stats[label, cv2.CC_STAT_WIDTH])
            h = int(stats[label, cv2.CC_STAT_HEIGHT])
            component_mask = np.where(labels == label, 255, 0).astype(np.uint8)
            contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue

            contour = max(contours, key=cv2.contourArea)
            centroid = (int(centroids[label][0]), int(centroids[label][1]))
            components.append(
                SegmentedComponent(
                    color_name=color_name,
                    mask=component_mask,
                    contour=contour,
                    area_px=area_px,
                    centroid_px=centroid,
                    bbox=(x, y, w, h),
                )
            )

    return components
