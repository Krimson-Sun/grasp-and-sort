from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import cv2
import message_filters
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Point
from manip_sort_interfaces.msg import DetectedObject, DetectedObjectArray, GraspCandidate
from rclpy.duration import Duration
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformException, TransformListener

from .debug_vis import draw_candidates, draw_components, draw_roi
from .geometry import (
    build_pose_from_world_point,
    compute_pca_angle,
    pixel_to_camera_point,
    transform_camera_point_to_world,
)
from .grasp_candidates import generate_candidates
from .segmentation import segment_components


def _parse_hsv_ranges(raw_ranges: Sequence[object]) -> List[Tuple[np.ndarray, np.ndarray]]:
    parsed: List[Tuple[np.ndarray, np.ndarray]] = []
    for entry in raw_ranges:
        if isinstance(entry, str):
            values = [int(part.strip()) for part in entry.split(",")]
        else:
            values = [int(part) for part in entry]
        if len(values) != 6:
            raise ValueError(f"Expected 6 HSV bounds values, got {len(values)} from {entry!r}")
        lower = np.array(values[0:3], dtype=np.uint8)
        upper = np.array(values[3:6], dtype=np.uint8)
        parsed.append((lower, upper))
    return parsed


class PerceptionGraspNode(Node):
    def __init__(self) -> None:
        super().__init__("perception_grasp_node")
        self._bridge = CvBridge()
        self._tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self.declare_parameter("camera.color_topic", "/camera/color/image_raw")
        self.declare_parameter("camera.depth_topic", "/camera/depth/image_raw")
        self.declare_parameter("camera.info_topic", "/camera/color/camera_info")
        self.declare_parameter("camera.frame", "camera_link")
        self.declare_parameter("world_frame", "world")
        self.declare_parameter("table_roi", [100, 50, 1180, 670])
        self.declare_parameter("table_z_world", 0.0)
        self.declare_parameter("grasp_z_offset", 0.01)
        self.declare_parameter("max_object_count", 8)
        self.declare_parameter("min_area", 1400)
        self.declare_parameter("morph_open_kernel", 5)
        self.declare_parameter("morph_close_kernel", 9)
        self.declare_parameter("min_clearance_px", 10.0)
        self.declare_parameter("grasp_width_min", 0.022)
        self.declare_parameter("grasp_width_max", 0.035)
        self.declare_parameter("support_length_min", 0.04)
        self.declare_parameter("max_local_depth_std", 0.006)
        self.declare_parameter("candidate_angle_delta_deg", 15.0)
        self.declare_parameter("score_weights", [0.30, 0.20, 0.20, 0.10, 0.10, 0.10])
        self.declare_parameter("debug.publish_overlay", True)
        self.declare_parameter("debug.publish_component_overlay", True)
        self.declare_parameter("debug.log_rejections", True)
        self.declare_parameter("debug.log_frame_summary", True)
        self.declare_parameter("debug.log_candidates", True)
        self.declare_parameter("debug.log_candidate_limit", 3)
        self.declare_parameter("color_order", ["red", "green", "blue", "yellow"])
        self.declare_parameter("color_to_class.red", "hammer")
        self.declare_parameter("color_to_class.green", "racket")
        self.declare_parameter("color_to_class.blue", "dumbbell")
        self.declare_parameter("color_to_class.yellow", "key")
        self.declare_parameter("color_to_bin.red", 1)
        self.declare_parameter("color_to_bin.green", 2)
        self.declare_parameter("color_to_bin.blue", 3)
        self.declare_parameter("color_to_bin.yellow", 4)
        self.declare_parameter("hsv_thresholds.red", ["0,120,70,10,255,255", "170,120,70,179,255,255"])
        self.declare_parameter("hsv_thresholds.green", ["35,70,60,90,255,255"])
        self.declare_parameter("hsv_thresholds.blue", ["95,90,60,130,255,255"])
        self.declare_parameter("hsv_thresholds.yellow", ["18,100,100,35,255,255"])

        self._camera_frame = self.get_parameter("camera.frame").value
        self._world_frame = self.get_parameter("world_frame").value
        self._table_roi = tuple(int(v) for v in self.get_parameter("table_roi").value)
        self._table_z_world = float(self.get_parameter("table_z_world").value)
        self._grasp_z_offset = float(self.get_parameter("grasp_z_offset").value)
        self._max_object_count = int(self.get_parameter("max_object_count").value)
        self._min_area = int(self.get_parameter("min_area").value)
        self._morph_open_kernel = int(self.get_parameter("morph_open_kernel").value)
        self._morph_close_kernel = int(self.get_parameter("morph_close_kernel").value)
        self._min_clearance_px = float(self.get_parameter("min_clearance_px").value)
        self._grasp_width_min = float(self.get_parameter("grasp_width_min").value)
        self._grasp_width_max = float(self.get_parameter("grasp_width_max").value)
        self._support_length_min = float(self.get_parameter("support_length_min").value)
        self._max_local_depth_std = float(self.get_parameter("max_local_depth_std").value)
        self._candidate_angle_delta_deg = float(self.get_parameter("candidate_angle_delta_deg").value)
        self._score_weights = [float(v) for v in self.get_parameter("score_weights").value]
        self._publish_overlay = bool(self.get_parameter("debug.publish_overlay").value)
        self._publish_component_overlay = bool(self.get_parameter("debug.publish_component_overlay").value)
        self._log_rejections = bool(self.get_parameter("debug.log_rejections").value)
        self._log_frame_summary = bool(self.get_parameter("debug.log_frame_summary").value)
        self._log_candidates = bool(self.get_parameter("debug.log_candidates").value)
        self._log_candidate_limit = int(self.get_parameter("debug.log_candidate_limit").value)

        color_order = [str(value) for value in self.get_parameter("color_order").value]
        self._color_to_class = {
            color: str(self.get_parameter(f"color_to_class.{color}").value)
            for color in color_order
        }
        self._color_to_bin = {
            color: int(self.get_parameter(f"color_to_bin.{color}").value)
            for color in color_order
        }
        self._hsv_thresholds = {
            color: _parse_hsv_ranges(self.get_parameter(f"hsv_thresholds.{color}").value)
            for color in color_order
        }

        self._objects_pub = self.create_publisher(DetectedObjectArray, "/detected_objects", 10)
        self._best_grasp_pub = self.create_publisher(DetectedObject, "/best_grasp", 10)
        self._overlay_pub = self.create_publisher(Image, "/debug/grasp_overlay", 10)
        self._components_pub = self.create_publisher(Image, "/debug/object_overlay", 10)

        color_topic = self.get_parameter("camera.color_topic").value
        depth_topic = self.get_parameter("camera.depth_topic").value
        info_topic = self.get_parameter("camera.info_topic").value
        color_sub = message_filters.Subscriber(self, Image, color_topic)
        depth_sub = message_filters.Subscriber(self, Image, depth_topic)
        info_sub = message_filters.Subscriber(self, CameraInfo, info_topic)
        self._sync = message_filters.ApproximateTimeSynchronizer(
            [color_sub, depth_sub, info_sub],
            queue_size=10,
            slop=0.2,
        )
        self._sync.registerCallback(self._on_images)
        self._frame_index = 0
        self._last_frame_wall_time = self.get_clock().now()
        self._camera_wait_timer = self.create_timer(5.0, self._log_camera_wait_state)
        self._log_info(
            "perception_grasp_node subscribed to "
            f"color='{color_topic}', depth='{depth_topic}', camera_info='{info_topic}', "
            f"camera_frame='{self._camera_frame}', world_frame='{self._world_frame}'"
        )

    def _log_info(self, message: str) -> None:
        self.get_logger().info(message)

    def _log_reject(self, reason: str, detail: str) -> None:
        if self._log_rejections:
            self._log_info(f"{reason}: {detail}")

    def _candidate_summary(self, candidate) -> str:
        yaw_deg = math.degrees(candidate.yaw)
        return (
            f"(u={candidate.u}, v={candidate.v}, yaw_deg={yaw_deg:.1f}, score={candidate.score:.3f}, "
            f"width={candidate.width:.3f}, support={candidate.support_length:.3f}, "
            f"flatness={candidate.flatness:.4f}, symmetry={candidate.symmetry:.3f})"
        )

    def _crop(self, image: np.ndarray) -> np.ndarray:
        x_min, y_min, x_max, y_max = self._table_roi
        return image[y_min:y_max, x_min:x_max]

    def _restore_point(self, u: int, v: int) -> Tuple[int, int]:
        x_min, y_min, _, _ = self._table_roi
        return u + x_min, v + y_min

    def _intrinsics_from_info(self, camera_info: CameraInfo) -> Sequence[float]:
        return [camera_info.k[0], camera_info.k[4], camera_info.k[2], camera_info.k[5]]

    def _lookup_camera_transform(self):
        return self._tf_buffer.lookup_transform(
            self._world_frame,
            self._camera_frame,
            rclpy.time.Time(),
            timeout=Duration(seconds=0.5),
        )

    def _log_camera_wait_state(self) -> None:
        seconds_since_last_frame = (
            self.get_clock().now() - self._last_frame_wall_time
        ).nanoseconds / 1e9
        if self._frame_index == 0:
            self.get_logger().warn(
                "waiting_for_camera_frames: no synchronized RGB/depth/camera_info callbacks yet on "
                f"'{self.get_parameter('camera.color_topic').value}', "
                f"'{self.get_parameter('camera.depth_topic').value}', "
                f"'{self.get_parameter('camera.info_topic').value}'"
            )
            return

        if seconds_since_last_frame > 2.0:
            self.get_logger().warn(
                f"camera_frame_gap: last synchronized frame arrived {seconds_since_last_frame:.1f}s ago"
            )

    def _on_images(self, color_msg: Image, depth_msg: Image, camera_info_msg: CameraInfo) -> None:
        self._frame_index += 1
        self._last_frame_wall_time = self.get_clock().now()
        try:
            transform = self._lookup_camera_transform()
        except TransformException as exc:
            self.get_logger().warn(f"camera_tf_unavailable: {exc}")
            return

        color_image = self._bridge.imgmsg_to_cv2(color_msg, desired_encoding="bgr8")
        depth_image = self._bridge.imgmsg_to_cv2(depth_msg)
        if depth_image.dtype != np.float32:
            depth_image = depth_image.astype(np.float32)
            if "16UC1" in depth_msg.encoding:
                depth_image *= 0.001

        color_roi = self._crop(color_image)
        depth_roi = self._crop(depth_image)
        components = segment_components(
            color_roi,
            self._hsv_thresholds,
            self._morph_open_kernel,
            self._morph_close_kernel,
            self._min_area,
        )
        intrinsics = self._intrinsics_from_info(camera_info_msg)
        if self._log_frame_summary:
            self._log_info(
                f"frame={self._frame_index} encoding(color={color_msg.encoding}, depth={depth_msg.encoding}) "
                f"roi={self._table_roi} segmented_components={len(components)}"
            )

        result = DetectedObjectArray()
        result.header = color_msg.header
        all_candidates_for_overlay = []
        best_object = None
        best_candidate_overlay = None

        for component in components[: self._max_object_count]:
            cx, cy = component.centroid_px
            if not (0 <= cy < depth_roi.shape[0] and 0 <= cx < depth_roi.shape[1]):
                self._log_reject("invalid_depth", f"{component.color_name}: centroid outside depth image")
                continue

            depth_samples = depth_roi[component.mask > 0]
            finite_samples = depth_samples[np.isfinite(depth_samples)]
            if finite_samples.size < 20:
                self._log_reject("invalid_depth", f"{component.color_name}: too few valid depth samples")
                continue

            centroid_depth = float(np.median(finite_samples))
            if self._log_frame_summary:
                self._log_info(
                    f"frame={self._frame_index} object_color={component.color_name} area_px={component.area_px} "
                    f"centroid_px={component.centroid_px} median_depth={centroid_depth:.3f}"
                )
            pca_angle = compute_pca_angle(component.mask)
            candidates = generate_candidates(
                component.mask,
                component.contour,
                depth_roi,
                component.centroid_px,
                pca_angle,
                intrinsics,
                self._candidate_angle_delta_deg,
                self._min_clearance_px,
                self._grasp_width_min,
                self._grasp_width_max,
                self._support_length_min,
                self._max_local_depth_std,
                self._score_weights,
            )
            if not candidates:
                self._log_reject("no_valid_grasp", component.color_name)
                continue
            if self._log_candidates:
                shortlist = "; ".join(
                    self._candidate_summary(candidate)
                    for candidate in candidates[: self._log_candidate_limit]
                )
                self._log_info(
                    f"frame={self._frame_index} object_color={component.color_name} "
                    f"candidate_count={len(candidates)} top_candidates={shortlist}"
                )

            detected = DetectedObject()
            detected.color_name = component.color_name
            detected.class_name = self._color_to_class[component.color_name]
            detected.object_id = detected.class_name
            detected.bin_id = self._color_to_bin[component.color_name]

            centroid_world = transform_camera_point_to_world(
                pixel_to_camera_point(cx + self._table_roi[0], cy + self._table_roi[1], centroid_depth, intrinsics),
                transform,
                self._camera_frame,
            )
            detected.centroid_world = Point(
                x=float(centroid_world[0]),
                y=float(centroid_world[1]),
                z=float(centroid_world[2]),
            )
            if self._log_frame_summary:
                self._log_info(
                    f"frame={self._frame_index} object_id={detected.object_id} "
                    f"centroid_world=({detected.centroid_world.x:.3f}, {detected.centroid_world.y:.3f}, {detected.centroid_world.z:.3f}) "
                    f"pca_deg={math.degrees(pca_angle):.1f}"
                )

            best_candidate = candidates[0]
            for candidate in candidates:
                global_u, global_v = self._restore_point(candidate.u, candidate.v)
                depth_m = float(depth_roi[candidate.v, candidate.u])
                camera_point = pixel_to_camera_point(global_u, global_v, depth_m, intrinsics)
                world_point = transform_camera_point_to_world(camera_point, transform, self._camera_frame)
                pose = build_pose_from_world_point(
                    world_point,
                    candidate.yaw,
                    z_override=world_point[2] + self._grasp_z_offset,
                    transform=transform,
                )

                candidate_msg = GraspCandidate()
                candidate_msg.pose = pose
                candidate_msg.score = float(candidate.score)
                candidate_msg.width = float(candidate.width)
                candidate_msg.support_length = float(candidate.support_length)
                candidate_msg.flatness = float(candidate.flatness)
                candidate_msg.symmetry = float(candidate.symmetry)
                detected.candidates.append(candidate_msg)
                all_candidates_for_overlay.append(
                    type(
                        "OverlayCandidate",
                        (),
                        {"u": global_u, "v": global_v, "yaw": candidate.yaw},
                    )()
                )

            detected.best_grasp_pose = detected.candidates[0].pose
            detected.best_grasp_score = detected.candidates[0].score
            result.objects.append(detected)
            if self._log_frame_summary:
                best_pose = detected.best_grasp_pose.position
                self._log_info(
                    f"frame={self._frame_index} selected_best_for_object={detected.object_id} "
                    f"best_score={detected.best_grasp_score:.3f} "
                    f"best_pose=({best_pose.x:.3f}, {best_pose.y:.3f}, {best_pose.z:.3f})"
                )

            if best_object is None or detected.best_grasp_score > best_object.best_grasp_score:
                best_object = detected
                best_candidate_overlay = type(
                    "OverlayCandidate",
                    (),
                    {
                        "u": self._restore_point(best_candidate.u, best_candidate.v)[0],
                        "v": self._restore_point(best_candidate.u, best_candidate.v)[1],
                        "yaw": best_candidate.yaw,
                    },
                )()

        result.objects.sort(
            key=lambda obj: math.hypot(obj.centroid_world.x, obj.centroid_world.y)
        )
        if self._log_frame_summary:
            ordered = ", ".join(
                f"{obj.object_id}@{math.hypot(obj.centroid_world.x, obj.centroid_world.y):.3f}"
                for obj in result.objects
            )
            self._log_info(
                f"frame={self._frame_index} published_objects={len(result.objects)} ordered_by_distance=[{ordered}]"
            )
        self._objects_pub.publish(result)
        if best_object is not None:
            self._best_grasp_pub.publish(best_object)

        if self._publish_component_overlay:
            component_overlay = draw_components(draw_roi(color_image, self._table_roi), components)
            self._components_pub.publish(self._bridge.cv2_to_imgmsg(component_overlay, encoding="bgr8"))

        if self._publish_overlay:
            overlay = draw_candidates(draw_roi(color_image, self._table_roi), all_candidates_for_overlay, best_candidate_overlay)
            self._overlay_pub.publish(self._bridge.cv2_to_imgmsg(overlay, encoding="bgr8"))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PerceptionGraspNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
