import cv2
import numpy as np

from manip_sort_perception.geometry import compute_pca_angle
from manip_sort_perception.grasp_candidates import generate_candidates


def test_grasp_candidates_prefer_dumbbell_neck():
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

    assert len(candidates) >= 1
    best = candidates[0]
    assert abs(best.u - 150) <= 18
    assert 100 <= best.v <= 200


def test_grasp_candidates_respect_flatness_threshold():
    mask = np.zeros((220, 220), dtype=np.uint8)
    cv2.rectangle(mask, (90, 40), (130, 180), 255, -1)
    depth = np.full(mask.shape, 1.0, dtype=np.float32)
    depth[80:140, 80:140] = np.linspace(0.97, 1.03, 60 * 60, dtype=np.float32).reshape(60, 60)

    candidates = generate_candidates(
        mask=mask,
        contour=np.array([]),
        depth_image_m=depth,
        centroid_px=(110, 110),
        base_angle=compute_pca_angle(mask),
        intrinsics=[700.0, 700.0, 110.0, 110.0],
        angle_delta_deg=15.0,
        min_clearance_px=8.0,
        grasp_width_min=0.022,
        grasp_width_max=0.040,
        support_length_min=0.04,
        max_local_depth_std=0.002,
        score_weights=[0.30, 0.20, 0.20, 0.10, 0.10, 0.10],
    )

    assert candidates == []
