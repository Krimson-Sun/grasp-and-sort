from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np
from geometry_msgs.msg import PointStamped, Pose
from tf2_geometry_msgs import do_transform_point


def compute_pca_angle(mask: np.ndarray) -> float:
    ys, xs = np.nonzero(mask > 0)
    if len(xs) < 2:
        return 0.0

    points = np.column_stack((xs.astype(np.float32), ys.astype(np.float32)))
    mean, eigenvectors = cv2.PCACompute(points, mean=None)
    del mean
    primary = eigenvectors[0]
    return float(math.atan2(primary[1], primary[0]))


def pixel_to_camera_point(u: float, v: float, depth_m: float, intrinsics: Sequence[float]) -> np.ndarray:
    fx, fy, cx, cy = intrinsics
    x = (u - cx) * depth_m / fx
    y = (v - cy) * depth_m / fy
    z = depth_m
    return np.array([x, y, z], dtype=np.float64)


def pixel_scale_m_per_px(depth_m: float, intrinsics: Sequence[float]) -> float:
    fx, fy, _, _ = intrinsics
    return float(depth_m * (1.0 / fx + 1.0 / fy) * 0.5)


def _quaternion_from_matrix(rotation: np.ndarray) -> Tuple[float, float, float, float]:
    trace = float(rotation[0, 0] + rotation[1, 1] + rotation[2, 2])
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (rotation[2, 1] - rotation[1, 2]) / s
        qy = (rotation[0, 2] - rotation[2, 0]) / s
        qz = (rotation[1, 0] - rotation[0, 1]) / s
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        s = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        qw = (rotation[2, 1] - rotation[1, 2]) / s
        qx = 0.25 * s
        qy = (rotation[0, 1] + rotation[1, 0]) / s
        qz = (rotation[0, 2] + rotation[2, 0]) / s
    elif rotation[1, 1] > rotation[2, 2]:
        s = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        qw = (rotation[0, 2] - rotation[2, 0]) / s
        qx = (rotation[0, 1] + rotation[1, 0]) / s
        qy = 0.25 * s
        qz = (rotation[1, 2] + rotation[2, 1]) / s
    else:
        s = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
        qw = (rotation[1, 0] - rotation[0, 1]) / s
        qx = (rotation[0, 2] + rotation[2, 0]) / s
        qy = (rotation[1, 2] + rotation[2, 1]) / s
        qz = 0.25 * s
    return (qx, qy, qz, qw)


def _rotation_matrix_from_quaternion(
    qx: float,
    qy: float,
    qz: float,
    qw: float,
) -> np.ndarray:
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def make_quaternion_from_axes(
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    z_axis: np.ndarray,
) -> Tuple[float, float, float, float]:
    rotation = np.column_stack(
        (
            x_axis / np.linalg.norm(x_axis),
            y_axis / np.linalg.norm(y_axis),
            z_axis / np.linalg.norm(z_axis),
        )
    )
    return _quaternion_from_matrix(rotation)


def rotate_camera_vector_to_world(vector_cam: np.ndarray, transform) -> np.ndarray:
    rotation = transform.transform.rotation
    matrix = _rotation_matrix_from_quaternion(
        rotation.x, rotation.y, rotation.z, rotation.w
    )
    return matrix @ vector_cam


def build_topdown_quaternion_from_camera_yaw(yaw: float, transform) -> Tuple[float, float, float, float]:
    # Candidate yaw is measured in camera optical coordinates:
    # +x right in the image, +y down in the image.
    support_cam = np.array([math.cos(yaw), math.sin(yaw), 0.0], dtype=np.float64)
    width_cam = np.array([-math.sin(yaw), math.cos(yaw), 0.0], dtype=np.float64)

    width_world = rotate_camera_vector_to_world(width_cam, transform)
    support_world_hint = rotate_camera_vector_to_world(support_cam, transform)

    width_world[2] = 0.0
    support_world_hint[2] = 0.0

    if np.linalg.norm(width_world) < 1e-9:
        width_world = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    else:
        width_world /= np.linalg.norm(width_world)

    approach_axis = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    support_world = np.cross(approach_axis, width_world)
    if np.linalg.norm(support_world) < 1e-9:
        support_world = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        support_world /= np.linalg.norm(support_world)

    if np.dot(support_world, support_world_hint) < 0.0:
        width_world = -width_world
        support_world = -support_world

    return make_quaternion_from_axes(approach_axis, width_world, support_world)


def yaw_to_topdown_quaternion(yaw: float) -> Tuple[float, float, float, float]:
    # Match the legacy demo convention: tool X points downward into the table.
    # Image yaw is measured in optical coordinates (x right, y down), so world Y flips sign.
    approach_axis = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    width_axis = np.array([math.sin(yaw), math.cos(yaw), 0.0], dtype=np.float64)
    support_axis = np.array([math.cos(yaw), -math.sin(yaw), 0.0], dtype=np.float64)
    return make_quaternion_from_axes(approach_axis, width_axis, support_axis)


def build_pose_from_world_point(
    world_xyz: np.ndarray,
    yaw: float,
    z_override: Optional[float] = None,
    transform=None,
) -> Pose:
    pose = Pose()
    pose.position.x = float(world_xyz[0])
    pose.position.y = float(world_xyz[1])
    pose.position.z = float(world_xyz[2] if z_override is None else z_override)
    if transform is None:
        qx, qy, qz, qw = yaw_to_topdown_quaternion(yaw)
    else:
        qx, qy, qz, qw = build_topdown_quaternion_from_camera_yaw(yaw, transform)
    pose.orientation.x = qx
    pose.orientation.y = qy
    pose.orientation.z = qz
    pose.orientation.w = qw
    return pose


def transform_camera_point_to_world(
    point_cam: np.ndarray,
    transform,
    camera_frame: str,
) -> np.ndarray:
    stamped = PointStamped()
    stamped.header.frame_id = camera_frame
    stamped.point.x = float(point_cam[0])
    stamped.point.y = float(point_cam[1])
    stamped.point.z = float(point_cam[2])
    transformed = do_transform_point(stamped, transform)
    return np.array(
        [transformed.point.x, transformed.point.y, transformed.point.z],
        dtype=np.float64,
    )
