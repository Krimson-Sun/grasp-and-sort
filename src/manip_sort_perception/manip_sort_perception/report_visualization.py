from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np

from .debug_vis import draw_candidates
from .geometry import compute_pca_angle
from .grasp_candidates import CandidateMetric, generate_candidates
from .segmentation import SegmentedComponent, segment_components
from .smoke_checks import make_synthetic_sort_scene


@dataclass
class ReportScene:
    name: str
    title: str
    image: np.ndarray
    depth: np.ndarray


@dataclass
class AnalyzedObject:
    component: SegmentedComponent
    candidates: List[CandidateMetric]

    @property
    def best_candidate(self) -> CandidateMetric:
        return self.candidates[0]


COLOR_TO_OBJECT = {
    "red": "hammer",
    "green": "racket",
    "blue": "dumbbell",
    "yellow": "key",
}


def _default_thresholds() -> Dict[str, List[Tuple[np.ndarray, np.ndarray]]]:
    def r(values: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
        return np.array(values[:3], dtype=np.uint8), np.array(values[3:6], dtype=np.uint8)

    return {
        "red": [r([0, 120, 70, 10, 255, 255]), r([170, 120, 70, 179, 255, 255])],
        "green": [r([35, 70, 60, 90, 255, 255])],
        "blue": [r([95, 90, 60, 130, 255, 255])],
        "yellow": [r([18, 100, 100, 35, 255, 255])],
    }


def _candidate_params(width: int, height: int) -> dict:
    return {
        "intrinsics": [1200.0, 1200.0, width * 0.5, height * 0.5],
        "angle_delta_deg": 15.0,
        "min_clearance_px": 10.0,
        "grasp_width_min": 0.022,
        "grasp_width_max": 0.035,
        "support_length_min": 0.04,
        "max_local_depth_std": 0.006,
        "score_weights": [0.30, 0.20, 0.20, 0.10, 0.10, 0.10],
    }


def _blank_scene(width: int = 1280, height: int = 720) -> tuple[np.ndarray, np.ndarray]:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:] = (38, 67, 107)
    cv2.rectangle(image, (90, 70), (1190, 650), (48, 80, 124), -1)
    cv2.rectangle(image, (115, 95), (1165, 625), (56, 93, 140), -1)
    depth = np.full((height, width), 1.02, dtype=np.float32)
    return image, depth


def _offset(center: Tuple[int, int], dx: float, dy: float, angle_deg: float) -> Tuple[int, int]:
    angle_rad = np.deg2rad(angle_deg)
    rx = dx * np.cos(angle_rad) - dy * np.sin(angle_rad)
    ry = dx * np.sin(angle_rad) + dy * np.cos(angle_rad)
    return int(round(center[0] + rx)), int(round(center[1] + ry))


def _draw_rotated_rect(
    image: np.ndarray,
    center: Tuple[int, int],
    size: Tuple[float, float],
    angle_deg: float,
    color: Tuple[int, int, int],
) -> None:
    rect = (tuple(map(float, center)), tuple(map(float, size)), float(angle_deg))
    box = cv2.boxPoints(rect).astype(np.int32)
    cv2.fillConvexPoly(image, box, color)


def _draw_hammer(
    image: np.ndarray,
    center: Tuple[int, int],
    angle_deg: float,
    color: Tuple[int, int, int],
    handle_width: int = 30,
    handle_length: int = 126,
    head_width: int = 118,
    head_height: int = 44,
) -> None:
    _draw_rotated_rect(image, center, (handle_width, handle_length), angle_deg, color)
    head_center = _offset(center, 0, -0.34 * handle_length, angle_deg)
    _draw_rotated_rect(image, head_center, (head_width, head_height), angle_deg, color)


def _draw_racket(
    image: np.ndarray,
    center: Tuple[int, int],
    angle_deg: float,
    color: Tuple[int, int, int],
    handle_width: int = 28,
    handle_length: int = 110,
    head_axes: Tuple[int, int] = (58, 48),
) -> None:
    _draw_rotated_rect(image, center, (handle_width, handle_length), angle_deg, color)
    head_center = _offset(center, 0, -0.54 * handle_length, angle_deg)
    cv2.ellipse(image, head_center, head_axes, angle_deg, 0, 360, color, -1)


def _draw_dumbbell(
    image: np.ndarray,
    center: Tuple[int, int],
    angle_deg: float,
    color: Tuple[int, int, int],
    neck_width: int = 30,
    neck_length: int = 112,
    bell_radius: int = 44,
) -> None:
    _draw_rotated_rect(image, center, (neck_width, neck_length), angle_deg, color)
    first = _offset(center, 0, -0.72 * neck_length, angle_deg)
    second = _offset(center, 0, 0.72 * neck_length, angle_deg)
    cv2.circle(image, first, bell_radius, color, -1)
    cv2.circle(image, second, bell_radius, color, -1)


def _draw_key(
    image: np.ndarray,
    center: Tuple[int, int],
    angle_deg: float,
    color: Tuple[int, int, int],
    stem_width: int = 28,
    stem_length: int = 124,
    head_radius: int = 38,
    tooth_size: Tuple[int, int] = (38, 32),
) -> None:
    _draw_rotated_rect(image, center, (stem_width, stem_length), angle_deg, color)
    head_center = _offset(center, 0, -0.56 * stem_length, angle_deg)
    tooth_center = _offset(center, 12, 0.46 * stem_length, angle_deg)
    cv2.circle(image, head_center, head_radius, color, -1)
    _draw_rotated_rect(image, tooth_center, tooth_size, angle_deg, color)


def _apply_depth_bump(
    depth: np.ndarray,
    center: Tuple[int, int],
    radius: int,
    amplitude: float = 0.012,
) -> None:
    yy, xx = np.ogrid[: depth.shape[0], : depth.shape[1]]
    dist_sq = (xx - center[0]) ** 2 + (yy - center[1]) ** 2
    mask = dist_sq <= radius * radius
    pattern = np.sin((xx - center[0]) * 0.22) + np.cos((yy - center[1]) * 0.19)
    depth[mask] += (amplitude * pattern[mask]).astype(np.float32)


def _build_report_scenes() -> List[ReportScene]:
    base_image, base_depth = make_synthetic_sort_scene()

    hammer_image, hammer_depth = _blank_scene()
    _draw_hammer(hammer_image, (305, 290), -6.0, (0, 0, 220), handle_width=30, handle_length=130)
    _draw_racket(hammer_image, (620, 290), 12.0, (0, 220, 0), handle_width=42, handle_length=100)
    _draw_dumbbell(hammer_image, (870, 300), 3.0, (220, 80, 0), neck_width=40, neck_length=110)
    _draw_key(hammer_image, (1060, 292), -12.0, (0, 220, 220), stem_width=24, stem_length=118)
    _apply_depth_bump(hammer_depth, (620, 290), 62, amplitude=0.011)

    key_image, key_depth = _blank_scene()
    _draw_hammer(key_image, (275, 300), 10.0, (0, 0, 220), handle_width=40, handle_length=116)
    _draw_racket(key_image, (560, 300), -8.0, (0, 220, 0), handle_width=36, handle_length=104)
    _draw_dumbbell(key_image, (815, 302), -10.0, (220, 80, 0), neck_width=40, neck_length=108)
    _draw_key(key_image, (1040, 285), 2.0, (0, 220, 220), stem_width=30, stem_length=132)
    _apply_depth_bump(key_depth, (560, 255), 56, amplitude=0.012)

    return [
        ReportScene("scene_01_balanced", "Scene 1: balanced arrangement", base_image, base_depth),
        ReportScene("scene_02_hammer_selected", "Scene 2: hammer selected", hammer_image, hammer_depth),
        ReportScene("scene_03_key_selected", "Scene 3: key selected", key_image, key_depth),
    ]


def _analyze_scene(scene: ReportScene) -> tuple[List[AnalyzedObject], AnalyzedObject]:
    params = _candidate_params(scene.image.shape[1], scene.image.shape[0])
    components = segment_components(
        scene.image,
        _default_thresholds(),
        morph_open_kernel=5,
        morph_close_kernel=9,
        min_area=1200,
    )
    analyzed: List[AnalyzedObject] = []
    for component in components:
        candidates = generate_candidates(
            mask=component.mask,
            contour=component.contour,
            depth_image_m=scene.depth,
            centroid_px=component.centroid_px,
            base_angle=compute_pca_angle(component.mask),
            **params,
        )
        if candidates:
            analyzed.append(AnalyzedObject(component=component, candidates=candidates))

    if not analyzed:
        raise RuntimeError(f"No valid grasp candidates generated for {scene.name}")

    selected = max(analyzed, key=lambda item: item.best_candidate.score)
    return analyzed, selected


def _put_panel_line(image: np.ndarray, text: str, origin: Tuple[int, int], color: Tuple[int, int, int]) -> None:
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        color,
        2,
        cv2.LINE_AA,
    )


def _draw_report_overlay(
    scene: ReportScene,
    analyzed: Iterable[AnalyzedObject],
    selected: AnalyzedObject,
) -> np.ndarray:
    output = scene.image.copy()
    tint = output.copy()
    tint[selected.component.mask > 0] = (155, 255, 190)
    output = cv2.addWeighted(output, 0.72, tint, 0.28, 0.0)

    all_candidates = []
    for item in analyzed:
        x, y, w, h = item.component.bbox
        contour_color = (210, 210, 210)
        line_width = 2
        if item is selected:
            contour_color = (30, 240, 120)
            line_width = 3
        cv2.drawContours(output, [item.component.contour], -1, contour_color, line_width)
        cv2.rectangle(output, (x, y), (x + w, y + h), contour_color, 1)
        object_name = COLOR_TO_OBJECT.get(item.component.color_name, item.component.color_name)
        label = object_name
        if item is selected:
            label += "  SELECTED"
        cv2.putText(
            output,
            label,
            (x, max(28, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            contour_color,
            2,
            cv2.LINE_AA,
        )
        if item is selected:
            all_candidates.extend(item.candidates)

    best = selected.best_candidate
    output = draw_candidates(output, all_candidates, best_candidate=best)
    cv2.circle(output, selected.component.centroid_px, 6, (255, 255, 255), -1)
    cv2.circle(output, selected.component.centroid_px, 11, (30, 240, 120), 2)

    panel = output.copy()
    cv2.rectangle(panel, (36, 24), (510, 172), (15, 18, 26), -1)
    output = cv2.addWeighted(output, 0.80, panel, 0.20, 0.0)
    cv2.rectangle(output, (36, 24), (510, 172), (220, 220, 220), 1)
    _put_panel_line(output, scene.title, (56, 60), (255, 255, 255))
    object_name = COLOR_TO_OBJECT.get(selected.component.color_name, selected.component.color_name)
    _put_panel_line(output, f"Selected object: {object_name}", (56, 96), (30, 240, 120))
    _put_panel_line(output, f"Candidates on object: {len(selected.candidates)}", (56, 132), (255, 255, 255))
    _put_panel_line(output, f"Best score: {best.score:.3f}", (56, 168), (255, 255, 255))
    return output


def _build_contact_sheet(frames: Sequence[np.ndarray]) -> np.ndarray:
    resized = [cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA) for frame in frames]
    gap = np.full((30, 640 * len(resized) + 20 * (len(resized) - 1), 3), 255, dtype=np.uint8)
    canvas = np.full((360, 640 * len(resized) + 20 * (len(resized) - 1), 3), 255, dtype=np.uint8)
    x = 0
    for frame in resized:
        canvas[:, x : x + frame.shape[1]] = frame
        x += frame.shape[1] + 20
    return np.vstack([canvas, gap])


def _crop_report_frame(image: np.ndarray, analyzed: Iterable[AnalyzedObject]) -> np.ndarray:
    boxes = [item.component.bbox for item in analyzed]
    x0 = min(36, min(x for x, _, _, _ in boxes) - 80)
    y0 = min(24, min(y for _, y, _, _ in boxes) - 90)
    x1 = max(max(x + w for x, _, w, _ in boxes) + 80, 520)
    y1 = max(max(y + h for _, y, _, h in boxes) + 70, 220)
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(image.shape[1], x1)
    y1 = min(image.shape[0], y1)
    return image[y0:y1, x0:x1]


def export_report_frames(output_dir: Path) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_paths: List[Path] = []
    rendered_frames: List[np.ndarray] = []
    for scene in _build_report_scenes():
        analyzed, selected = _analyze_scene(scene)
        rendered = _draw_report_overlay(scene, analyzed, selected)
        rendered = _crop_report_frame(rendered, analyzed)
        rendered_frames.append(rendered)
        output_path = output_dir / f"{scene.name}.png"
        cv2.imwrite(str(output_path), rendered)
        generated_paths.append(output_path)
        print(
            f"{output_path.name}: selected={COLOR_TO_OBJECT.get(selected.component.color_name, selected.component.color_name)}, "
            f"score={selected.best_candidate.score:.3f}, candidates={len(selected.candidates)}"
        )

    contact_sheet = _build_contact_sheet(rendered_frames)
    contact_sheet_path = output_dir / "scene_contact_sheet.png"
    cv2.imwrite(str(contact_sheet_path), contact_sheet)
    generated_paths.append(contact_sheet_path)
    print(f"{contact_sheet_path.name}: generated contact sheet")
    return generated_paths


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate report-ready perception/grasp visualizations.")
    parser.add_argument(
        "--output-dir",
        default="report_assets/perception_grasp",
        help="Directory where PNG frames will be written.",
    )
    args = parser.parse_args(argv)
    export_report_frames(Path(args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
