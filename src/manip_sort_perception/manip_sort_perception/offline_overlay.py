from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import yaml

from .debug_vis import draw_candidates
from .geometry import compute_pca_angle
from .grasp_candidates import CandidateMetric, generate_candidates
from .segmentation import SegmentedComponent, segment_components


COLOR_TO_OBJECT = {
    "red": "hammer",
    "green": "racket",
    "blue": "dumbbell",
    "yellow": "key",
}


@dataclass
class OverlayConfig:
    table_roi: Tuple[int, int, int, int]
    morph_open_kernel: int
    morph_close_kernel: int
    min_area: int
    min_clearance_px: float
    grasp_width_min: float
    grasp_width_max: float
    support_length_min: float
    max_local_depth_std: float
    candidate_angle_delta_deg: float
    score_weights: List[float]
    hsv_thresholds: Dict[str, List[Tuple[np.ndarray, np.ndarray]]]


@dataclass
class AnalyzedObject:
    component_global: SegmentedComponent
    candidates_global: List[CandidateMetric]

    @property
    def best_candidate(self) -> CandidateMetric:
        return self.candidates_global[0]

    @property
    def object_name(self) -> str:
        return COLOR_TO_OBJECT.get(self.component_global.color_name, self.component_global.color_name)


def _parse_hsv_ranges(raw_ranges: Sequence[object]) -> List[Tuple[np.ndarray, np.ndarray]]:
    parsed: List[Tuple[np.ndarray, np.ndarray]] = []
    for entry in raw_ranges:
        if isinstance(entry, str):
            values = [int(part.strip()) for part in entry.split(",")]
        else:
            values = [int(part) for part in entry]
        lower = np.array(values[0:3], dtype=np.uint8)
        upper = np.array(values[3:6], dtype=np.uint8)
        parsed.append((lower, upper))
    return parsed


def load_overlay_config(config_path: Path) -> OverlayConfig:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    params = payload["perception_grasp_node"]["ros__parameters"]
    hsv_thresholds = {
        color: _parse_hsv_ranges(ranges)
        for color, ranges in params["hsv_thresholds"].items()
    }
    return OverlayConfig(
        table_roi=tuple(int(value) for value in params["table_roi"]),
        morph_open_kernel=int(params["morph_open_kernel"]),
        morph_close_kernel=int(params["morph_close_kernel"]),
        min_area=int(params["min_area"]),
        min_clearance_px=float(params["min_clearance_px"]),
        grasp_width_min=float(params["grasp_width_min"]),
        grasp_width_max=float(params["grasp_width_max"]),
        support_length_min=float(params["support_length_min"]),
        max_local_depth_std=float(params["max_local_depth_std"]),
        candidate_angle_delta_deg=float(params["candidate_angle_delta_deg"]),
        score_weights=[float(value) for value in params["score_weights"]],
        hsv_thresholds=hsv_thresholds,
    )


def _load_input_paths(inputs: Iterable[str]) -> List[Path]:
    resolved: List[Path] = []
    for raw_input in inputs:
        candidate = Path(raw_input)
        if candidate.is_dir():
            for pattern in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"):
                resolved.extend(sorted(candidate.glob(pattern)))
            continue
        if any(char in raw_input for char in "*?[]"):
            resolved.extend(sorted(Path().glob(raw_input)))
            continue
        resolved.append(candidate)
    unique = []
    seen = set()
    for path in resolved:
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)
    return unique


def _full_frame_roi(image: np.ndarray) -> Tuple[int, int, int, int]:
    return (0, 0, image.shape[1], image.shape[0])


def _clip_roi(roi: Tuple[int, int, int, int], image: np.ndarray) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = roi
    x0 = max(0, min(image.shape[1] - 1, x0))
    y0 = max(0, min(image.shape[0] - 1, y0))
    x1 = max(x0 + 1, min(image.shape[1], x1))
    y1 = max(y0 + 1, min(image.shape[0], y1))
    return (x0, y0, x1, y1)


def _translate_component(component: SegmentedComponent, dx: int, dy: int) -> SegmentedComponent:
    contour = component.contour.copy()
    contour[:, 0, 0] += dx
    contour[:, 0, 1] += dy
    x, y, w, h = component.bbox
    return SegmentedComponent(
        color_name=component.color_name,
        mask=component.mask.copy(),
        contour=contour,
        area_px=component.area_px,
        centroid_px=(component.centroid_px[0] + dx, component.centroid_px[1] + dy),
        bbox=(x + dx, y + dy, w, h),
    )


def _translate_candidates(candidates: Sequence[CandidateMetric], dx: int, dy: int) -> List[CandidateMetric]:
    translated: List[CandidateMetric] = []
    for candidate in candidates:
        translated.append(
            CandidateMetric(
                u=candidate.u + dx,
                v=candidate.v + dy,
                yaw=candidate.yaw,
                score=candidate.score,
                width=candidate.width,
                support_length=candidate.support_length,
                flatness=candidate.flatness,
                symmetry=candidate.symmetry,
            )
        )
    return translated


def analyze_image(
    image: np.ndarray,
    config: OverlayConfig,
    use_roi: bool,
    assumed_depth_m: float,
    focal_length_px: float,
) -> tuple[List[AnalyzedObject], AnalyzedObject, Tuple[int, int, int, int]]:
    roi = _clip_roi(config.table_roi, image) if use_roi else _full_frame_roi(image)
    x0, y0, x1, y1 = roi
    color_roi = image[y0:y1, x0:x1]
    depth_roi = np.full(color_roi.shape[:2], assumed_depth_m, dtype=np.float32)
    cx = color_roi.shape[1] * 0.5
    cy = color_roi.shape[0] * 0.5
    intrinsics = [focal_length_px, focal_length_px, cx, cy]

    components = segment_components(
        color_roi,
        config.hsv_thresholds,
        morph_open_kernel=config.morph_open_kernel,
        morph_close_kernel=config.morph_close_kernel,
        min_area=config.min_area,
    )

    analyzed: List[AnalyzedObject] = []
    for component in components:
        candidates = generate_candidates(
            mask=component.mask,
            contour=component.contour,
            depth_image_m=depth_roi,
            centroid_px=component.centroid_px,
            base_angle=compute_pca_angle(component.mask),
            intrinsics=intrinsics,
            angle_delta_deg=config.candidate_angle_delta_deg,
            min_clearance_px=config.min_clearance_px,
            grasp_width_min=config.grasp_width_min,
            grasp_width_max=config.grasp_width_max,
            support_length_min=config.support_length_min,
            max_local_depth_std=config.max_local_depth_std,
            score_weights=config.score_weights,
        )
        if not candidates:
            continue
        analyzed.append(
            AnalyzedObject(
                component_global=_translate_component(component, x0, y0),
                candidates_global=_translate_candidates(candidates, x0, y0),
            )
        )

    if not analyzed:
        raise RuntimeError("No valid objects with grasp candidates were found")

    selected = max(analyzed, key=lambda item: item.best_candidate.score)
    return analyzed, selected, roi


def _put_text(image: np.ndarray, text: str, xy: Tuple[int, int], color: Tuple[int, int, int], scale: float = 0.72) -> None:
    cv2.putText(image, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def render_overlay(
    image: np.ndarray,
    analyzed: Sequence[AnalyzedObject],
    selected: AnalyzedObject,
    roi: Tuple[int, int, int, int],
) -> np.ndarray:
    output = image.copy()

    roi_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    x0, y0, x1, y1 = roi
    roi_mask[y0:y1, x0:x1] = 255
    shaded = output.copy()
    shaded[roi_mask == 0] = (shaded[roi_mask == 0] * 0.45).astype(np.uint8)
    output = shaded
    cv2.rectangle(output, (x0, y0), (x1, y1), (220, 220, 220), 2)

    highlight = output.copy()
    selected_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(selected_mask, [selected.component_global.contour], -1, 255, -1)
    highlight[selected_mask > 0] = (155, 255, 190)
    output = cv2.addWeighted(output, 0.76, highlight, 0.24, 0.0)

    for item in analyzed:
        x, y, w, h = item.component_global.bbox
        color = (225, 225, 225)
        thickness = 1
        if item is selected:
            color = (30, 240, 120)
            thickness = 3
        cv2.drawContours(output, [item.component_global.contour], -1, color, thickness)
        cv2.rectangle(output, (x, y), (x + w, y + h), color, 1)
        label = item.object_name
        if item is selected:
            label += " SELECTED"
        _put_text(output, label, (x, max(28, y - 10)), color, scale=0.58)

    output = draw_candidates(output, selected.candidates_global, best_candidate=selected.best_candidate)
    cv2.circle(output, selected.component_global.centroid_px, 6, (255, 255, 255), -1)
    cv2.circle(output, selected.component_global.centroid_px, 12, (30, 240, 120), 2)

    panel = output.copy()
    cv2.rectangle(panel, (34, 24), (470, 156), (15, 18, 26), -1)
    output = cv2.addWeighted(output, 0.82, panel, 0.18, 0.0)
    cv2.rectangle(output, (34, 24), (470, 156), (220, 220, 220), 1)
    _put_text(output, f"Selected object: {selected.object_name}", (54, 60), (30, 240, 120))
    _put_text(output, f"Candidates on object: {len(selected.candidates_global)}", (54, 95), (255, 255, 255))
    _put_text(output, f"Best score: {selected.best_candidate.score:.3f}", (54, 130), (255, 255, 255))
    return output


def process_images(
    input_paths: Sequence[Path],
    output_dir: Path,
    config: OverlayConfig,
    use_roi: bool,
    assumed_depth_m: float,
    focal_length_px: float,
) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: List[Path] = []
    for input_path in input_paths:
        image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read image: {input_path}")
        analyzed, selected, roi = analyze_image(
            image=image,
            config=config,
            use_roi=use_roi,
            assumed_depth_m=assumed_depth_m,
            focal_length_px=focal_length_px,
        )
        rendered = render_overlay(image, analyzed, selected, roi)
        output_path = output_dir / f"{input_path.stem}_overlay.png"
        cv2.imwrite(str(output_path), rendered)
        generated.append(output_path)
        print(
            f"{output_path.name}: selected={selected.object_name}, "
            f"score={selected.best_candidate.score:.3f}, candidates={len(selected.candidates_global)}"
        )
    return generated


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create grasp-selection overlays for real scene photos.")
    parser.add_argument("inputs", nargs="+", help="Input image files, directories, or glob patterns.")
    parser.add_argument(
        "--output-dir",
        default="report_assets/real_scene_overlays",
        help="Directory for rendered overlay PNG files.",
    )
    parser.add_argument(
        "--config",
        default="src/manip_sort_perception/config/perception.yaml",
        help="Path to perception YAML with HSV/ROI settings.",
    )
    parser.add_argument(
        "--no-roi",
        action="store_true",
        help="Process the whole image instead of the configured table ROI.",
    )
    parser.add_argument(
        "--assumed-depth-m",
        type=float,
        default=1.0,
        help="Approximate scene depth in meters used for candidate sizing.",
    )
    parser.add_argument(
        "--focal-length-px",
        type=float,
        default=1200.0,
        help="Approximate focal length in pixels.",
    )
    args = parser.parse_args(argv)

    input_paths = _load_input_paths(args.inputs)
    if not input_paths:
        raise SystemExit("No input images found.")

    process_images(
        input_paths=input_paths,
        output_dir=Path(args.output_dir),
        config=load_overlay_config(Path(args.config)),
        use_roi=not args.no_roi,
        assumed_depth_m=args.assumed_depth_m,
        focal_length_px=args.focal_length_px,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
