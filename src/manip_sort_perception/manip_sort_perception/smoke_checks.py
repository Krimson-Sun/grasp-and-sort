from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

from .geometry import compute_pca_angle
from .grasp_candidates import generate_candidates
from .segmentation import segment_components


@dataclass
class SmokeResult:
    name: str
    passed: bool
    detail: str


def _default_thresholds() -> Dict[str, List[Tuple[np.ndarray, np.ndarray]]]:
    def r(values: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
        return np.array(values[:3], dtype=np.uint8), np.array(values[3:6], dtype=np.uint8)

    return {
        "red": [r([0, 120, 70, 10, 255, 255]), r([170, 120, 70, 179, 255, 255])],
        "green": [r([35, 70, 60, 90, 255, 255])],
        "blue": [r([95, 90, 60, 130, 255, 255])],
        "yellow": [r([18, 100, 100, 35, 255, 255])],
    }


def make_synthetic_sort_scene(width: int = 1280, height: int = 720) -> tuple[np.ndarray, np.ndarray]:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:] = (40, 70, 110)

    cv2.rectangle(image, (250, 150), (300, 320), (0, 0, 220), -1)
    cv2.rectangle(image, (220, 150), (330, 210), (0, 0, 220), -1)

    cv2.rectangle(image, (500, 170), (545, 300), (0, 220, 0), -1)
    cv2.ellipse(image, (522, 150), (65, 52), 0, 0, 360, (0, 220, 0), -1)

    cv2.rectangle(image, (760, 190), (805, 325), (220, 80, 0), -1)
    cv2.circle(image, (782, 165), 45, (220, 80, 0), -1)
    cv2.circle(image, (782, 350), 45, (220, 80, 0), -1)

    cv2.rectangle(image, (980, 180), (1012, 335), (0, 220, 220), -1)
    cv2.circle(image, (996, 145), 42, (0, 220, 220), -1)
    cv2.rectangle(image, (1000, 315), (1035, 360), (0, 220, 220), -1)

    depth = np.full((height, width), 1.02, dtype=np.float32)
    return image, depth


def run_segmentation_smoke_check() -> SmokeResult:
    image, _ = make_synthetic_sort_scene()
    components = segment_components(
        image,
        _default_thresholds(),
        morph_open_kernel=5,
        morph_close_kernel=9,
        min_area=1200,
    )
    colors = sorted(component.color_name for component in components)
    passed = colors == ["blue", "green", "red", "yellow"]
    return SmokeResult("segmentation", passed, f"detected_colors={colors}")


def run_grasp_smoke_check() -> SmokeResult:
    mask = np.zeros((300, 300), dtype=np.uint8)
    cv2.rectangle(mask, (132, 95), (168, 205), 255, -1)
    cv2.circle(mask, (150, 75), 40, 255, -1)
    cv2.circle(mask, (150, 225), 40, 255, -1)

    depth = np.full(mask.shape, 1.0, dtype=np.float32)
    candidates = generate_candidates(
        mask=mask,
        contour=np.array([]),
        depth_image_m=depth,
        centroid_px=(150, 150),
        base_angle=compute_pca_angle(mask),
        intrinsics=[1200.0, 1200.0, 150.0, 150.0],
        angle_delta_deg=15.0,
        min_clearance_px=10.0,
        grasp_width_min=0.022,
        grasp_width_max=0.040,
        support_length_min=0.04,
        max_local_depth_std=0.005,
        score_weights=[0.30, 0.20, 0.20, 0.10, 0.10, 0.10],
    )
    if not candidates:
        return SmokeResult("grasp_candidates", False, "no candidates generated")

    best = candidates[0]
    passed = abs(best.u - 150) <= 18 and 100 <= best.v <= 200
    detail = f"best=(u={best.u},v={best.v},score={best.score:.3f},width={best.width:.3f})"
    return SmokeResult("grasp_candidates", passed, detail)


def run_all_smoke_checks() -> List[SmokeResult]:
    return [run_segmentation_smoke_check(), run_grasp_smoke_check()]


def main() -> int:
    results = run_all_smoke_checks()
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"[{status}] {result.name}: {result.detail}")
    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
