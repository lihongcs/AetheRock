import dataclasses
import importlib.util
import json
import logging
import os
from pathlib import Path
import queue
import shutil
import signal as signal_module
import socket
import subprocess
import sys
import threading
import time
from typing import Any, Callable

import cv2
import numpy as np
import scipy.spatial.transform as st

from openpi_client import image_tools


_REPO_ROOT = Path(__file__).resolve().parents[1]
_LOCAL_LEROBOT_SRC = _REPO_ROOT / "src" / "openpi"
if str(_LOCAL_LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(_LOCAL_LEROBOT_SRC))


def _load_local_pose_utils() -> Any:
    pose_utils_path = _LOCAL_LEROBOT_SRC / "lerobot" / "datasets" / "pose_utils.py"
    spec = importlib.util.spec_from_file_location("_openpi_local_pose_utils", pose_utils_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load pose_utils from {pose_utils_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_POSE_UTILS = _load_local_pose_utils()
mat_to_certain_pose_type = _POSE_UTILS.mat_to_certain_pose_type
mat_to_pose = _POSE_UTILS.mat_to_pose
pose_to_mat = _POSE_UTILS.pose_to_mat
pose_to_pos_rot = _POSE_UTILS.pose_to_pos_rot
pos_rot_to_mat = _POSE_UTILS.pos_rot_to_mat


LOGGER = logging.getLogger(__name__)
_ACTIVE_RECORDER = None
_SHUTTING_DOWN = False
def _env_shape(name: str, default: tuple[int, int]) -> tuple[int, int]:
    value = os.environ.get(name)
    if not value:
        return default
    parts = value.lower().replace("x", ",").split(",")
    if len(parts) != 2:
        raise RuntimeError(f"{name} must be WIDTHxHEIGHT or WIDTH,HEIGHT, got {value!r}")
    return int(parts[0]), int(parts[1])


WRIST_CAPTURE_SHAPE = _env_shape("HANDCAP_WRIST_CAPTURE_SHAPE", (768, 768))
WRIST_MODEL_INPUT_SHAPE = _env_shape("HANDCAP_WRIST_MODEL_INPUT_SHAPE", (224, 224))
WRIST_RECORD_SHAPE = _env_shape("HANDCAP_WRIST_RECORD_SHAPE", WRIST_CAPTURE_SHAPE)
WRIST_PREPROCESS_MODE = os.environ.get("HANDCAP_WRIST_PREPROCESS", "mvs_legacy").strip().lower()
MODEL_INPUT_HIGHLIGHT_THICKNESS = 16


def _repo_root() -> Path:
    return _REPO_ROOT


def _default_camera_config_path() -> Path:
    return _repo_root() / "src" / "perception" / "configs" / "camera" / "handcap_realsense_tactile.json"


def _load_base_camera() -> Any:
    from perception.cameras.base_camera import BaseCamera
    return BaseCamera


def _required_env_float(name: str) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        raise RuntimeError(f"{name} must be set by the launch script before starting real-robot inference.")
    try:
        return float(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a float, got {value!r}. Set it in run_inference_pi05.sh.") from exc


def _required_env_bool(name: str) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        raise RuntimeError(f"{name} must be set by the launch script before starting real-robot inference.")
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"{name} must be one of 1/0/true/false/yes/no/on/off, got {value!r}.")


def _pose7_to_mat(pose7: Any) -> np.ndarray:
    pose = np.asarray(pose7, dtype=np.float64)
    pos = pose[:3]
    qw, qx, qy, qz = pose[3:]
    return pos_rot_to_mat(pos, st.Rotation.from_quat([qx, qy, qz, qw]))


def _mat_to_tcp_pose7(mat: np.ndarray) -> list[float]:
    pos = np.asarray(mat[:3, 3], dtype=np.float64)
    quat_xyzw = st.Rotation.from_matrix(mat[:3, :3]).as_quat(scalar_first=False)
    return [
        float(pos[0]),
        float(pos[1]),
        float(pos[2]),
        float(quat_xyzw[3]),
        float(quat_xyzw[0]),
        float(quat_xyzw[1]),
        float(quat_xyzw[2]),
    ]


from scipy.spatial.transform import Rotation

def encode_state_to_handcap_state(eepose_rotvec: np.ndarray, gripper_width: float) -> np.ndarray:
    # Because the dataset's __getitem__ processes all states into RELATIVE states using process_to_relative_rot6d,
    # the model was trained expecting the current observation.state to be the origin (Identity matrix).
    # Therefore, we MUST feed it the identity pose (pos=0, rot=identity) during inference!
    identity_pos = np.array([0.0, 0.0, 0.0])
    identity_rot6d = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    pose10d = np.concatenate([identity_pos, identity_rot6d])
    return np.concatenate([pose10d, np.array([gripper_width], dtype=np.float64)], axis=-1)

def get_real_umi_inference_action(raw_action_rotvec: np.ndarray, abs_pose_rotvec: np.ndarray) -> np.ndarray:
    # raw_action_rotvec is shape (6,) -> [pos, rotvec]
    # abs_pose_rotvec is shape (7,) -> [pos, rotvec, gripper] or (6,) -> [pos, rotvec]
    
    current_pose_mat = pose_to_mat(abs_pose_rotvec[:6])
    relative_action_mat = pose_to_mat(raw_action_rotvec[:6])
    
    # target_pose = current_pose @ relative_action
    target_pose_mat = current_pose_mat @ relative_action_mat
    
    target_pose_rotvec = mat_to_certain_pose_type(target_pose_mat, "rotvec")
    return target_pose_rotvec

def _resize_image_area(image: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    target_w, target_h = target_shape
    if image.shape[1] == target_w and image.shape[0] == target_h:
        return image
    return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)


def _resize_image_high_quality(image: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    target_w, target_h = target_shape
    h, w = image.shape[:2]
    if (h, w) == (target_h, target_w):
        return image
    interpolation = cv2.INTER_AREA if target_w < w or target_h < h else cv2.INTER_LANCZOS4
    return cv2.resize(image, (target_w, target_h), interpolation=interpolation)


def _prepare_wrist_like_collection(image_bgr: np.ndarray) -> np.ndarray:
    if WRIST_PREPROCESS_MODE in {"lerobot_224", "none"}:
        return np.ascontiguousarray(image_bgr)
    # Legacy MVS path: match handcap_rgb.py by normalizing raw MVS wrist frames.
    return _resize_image_area(image_bgr, WRIST_CAPTURE_SHAPE)


def decode_handcap_action_chunk(action_chunk: np.ndarray) -> np.ndarray:
    actions = np.asarray(action_chunk, dtype=np.float64)
    if actions.ndim != 2:
        raise ValueError(f"Expected action chunk with rank 2, got shape {actions.shape}")
    if actions.shape[-1] < 10:
        raise ValueError(f"Expected handcap action dim >= 10, got {actions.shape[-1]}")
    actions = actions[:, :10]

    pose_mat = _pose10d_to_mat(actions[:, :9])
    pose_rotvec = mat_to_certain_pose_type(pose_mat, "rotvec")
    return np.concatenate([pose_rotvec, actions[:, 9:10]], axis=-1)


def _pose10d_to_mat(d10: np.ndarray) -> np.ndarray:
    pos = d10[..., :3]
    rot_mat = _rot6d_to_mat(d10[..., 3:])
    mat = np.zeros(d10.shape[:-1] + (4, 4), dtype=d10.dtype)
    mat[..., :3, 3] = pos
    mat[..., :3, :3] = rot_mat
    mat[..., 3, 3] = 1
    return mat


def _rot6d_to_mat(d6: np.ndarray) -> np.ndarray:
    a1 = d6[..., :3]
    a2 = d6[..., 3:]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2, axis=-1)
    return np.stack((b1, b2, b3), axis=-2)


def _prepare_mvs_image(image_bgr: np.ndarray) -> np.ndarray:
    """严格且唯一的 MVS 相机处理逻辑，保证与数采和离线处理 100% 对齐，不允许出错"""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    # 1. 模拟 handcap_rgb.py: 1680x1680 -> 768x768 (cv2.INTER_AREA)
    rgb = cv2.resize(rgb, (768, 768), interpolation=cv2.INTER_AREA)
    
    # 2. 模拟 _01_combine...: 768x768 -> 224x224 (cv2.INTER_AREA)
    rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
    
    return image_tools.convert_to_uint8(rgb)


def _prepare_realsense_image(image_bgr: np.ndarray) -> np.ndarray:
    """单独的 RealSense 相机处理逻辑，处理成一样的高 (224)，维持原生宽高比 (更宽)"""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    cur_h, cur_w = rgb.shape[:2]
    
    target_h = 224
    scale = target_h / cur_h
    target_w = max(1, int(round(cur_w * scale)))
    
    rgb = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)
    return image_tools.convert_to_uint8(rgb)


def _prepare_camera_image(image_bgr: np.ndarray) -> np.ndarray:
    if WRIST_PREPROCESS_MODE in {"lerobot_224", "none"}:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        if WRIST_PREPROCESS_MODE == "lerobot_224" and rgb.shape[:2] != (224, 224):
            rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
        return image_tools.convert_to_uint8(rgb)

    # Legacy dispatcher for MVS/RealSense configs.
    cur_h, cur_w = image_bgr.shape[:2]
    if cur_h == cur_w:
        return _prepare_mvs_image(image_bgr)
    else:
        return _prepare_realsense_image(image_bgr)


def _prepare_tactile_image(image_bgr: np.ndarray | None, use_tactile: bool) -> np.ndarray:
    if image_bgr is None:
        if use_tactile:
            raise ValueError("Tactile image is missing while use_tactile=True")
        return np.zeros((224, 224, 3), dtype=np.uint8)

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    if rgb.shape[:2] != (224, 224):
        rgb = _resize_image_high_quality(rgb, (224, 224))
    rgb = image_tools.convert_to_uint8(rgb)
    if use_tactile:
        return rgb
    return np.zeros_like(rgb)


def _camera_get_bgr(camera) -> tuple[np.ndarray | None, float | None]:
    try:
        image, timestamp = camera.get_data(return_type="bgr")
    except TypeError:
        image, timestamp = camera.get_data()
    if image is None:
        return None, timestamp
    image = np.asarray(image)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    elif image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Unsupported tactile frame shape: {image.shape}")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image), timestamp


def _line_intersection(line1, line2):
    x1, y1, x2, y2 = line1[0][0], line1[0][1], line1[1][0], line1[1][1]
    x3, y3, x4, y4 = line2[0][0], line2[0][1], line2[1][0], line2[1][1]
    det1 = x1 * y2 - y1 * x2
    det2 = x3 * y4 - y3 * x4
    dx1, dy1 = x1 - x2, y1 - y2
    dx2, dy2 = x3 - x4, y3 - y4
    denom = (dx1 * dy2) - (dy1 * dx2)
    if denom == 0:
        return None
    px = (det1 * dx2 - dx1 * det2) / denom
    py = (det1 * dy2 - dy1 * det2) / denom
    return (px, py)


def _compute_tactile_calibration_params(reference_image_bgr: np.ndarray):
    height, width = reference_image_bgr.shape[:2]
    gray = cv2.cvtColor(reference_image_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found in tactile reference image")

    largest_contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(largest_contour)
    epsilon = 0.02 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)
    pts = approx.reshape(-1, 2)
    valid_lines = []
    margin = 5

    for i in range(len(pts)):
        pt1 = pts[i]
        pt2 = pts[(i + 1) % len(pts)]
        if abs(pt1[0] - pt2[0]) < margin and (pt1[0] <= margin or pt1[0] >= width - margin):
            continue
        if abs(pt1[1] - pt2[1]) < margin and (pt1[1] <= margin or pt1[1] >= height - margin):
            continue
        on_border = False
        for coord, max_val in [(0, width), (1, height)]:
            if (pt1[coord] <= margin and pt2[coord] <= margin) or (
                pt1[coord] >= max_val - margin and pt2[coord] >= max_val - margin
            ):
                on_border = True
                break
        if not on_border:
            valid_lines.append((pt1, pt2))

    if len(valid_lines) < 6:
        raise ValueError(f"Could not extract 6 valid tactile calibration lines; found {len(valid_lines)}")
    if len(valid_lines) > 6:
        line_lengths = [np.hypot(p1[0] - p2[0], p1[1] - p2[1]) for p1, p2 in valid_lines]
        longest_indices = sorted(np.argsort(line_lengths)[-6:])
        valid_lines = [valid_lines[i] for i in longest_indices]

    true_vertices = []
    for i in range(6):
        intersect = _line_intersection(valid_lines[i], valid_lines[(i + 1) % 6])
        if intersect is None:
            raise ValueError("Adjacent tactile calibration lines are parallel")
        true_vertices.append(intersect)

    true_vertices = np.array(true_vertices)
    cx = np.mean(true_vertices[:, 0])
    cy = np.mean(true_vertices[:, 1])
    angles = np.arctan2(true_vertices[:, 1] - cy, true_vertices[:, 0] - cx)
    sorted_vertices = true_vertices[np.argsort(angles)]

    target_l = min(width / np.sqrt(3), height / 2.0)
    target_cx = width / 2.0
    target_cy = height / 2.0
    ideal_pts = np.array(
        [
            [target_cx - target_l * np.sqrt(3) / 2, target_cy - target_l / 2],
            [target_cx, target_cy - target_l],
            [target_cx + target_l * np.sqrt(3) / 2, target_cy - target_l / 2],
            [target_cx + target_l * np.sqrt(3) / 2, target_cy + target_l / 2],
            [target_cx, target_cy + target_l],
            [target_cx - target_l * np.sqrt(3) / 2, target_cy + target_l / 2],
        ],
        dtype=np.float32,
    )

    homography, _ = cv2.findHomography(np.array(sorted_vertices, dtype=np.float32), ideal_pts)
    ideal_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(ideal_mask, [np.int32(ideal_pts)], 255)
    return homography, ideal_mask, width, height


def _apply_tactile_calibration(image_bgr: np.ndarray, params, interpolation=cv2.INTER_LANCZOS4) -> np.ndarray:
    homography, ideal_mask, out_w, out_h = params
    warped = cv2.warpPerspective(
        image_bgr,
        homography,
        (out_w, out_h),
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return cv2.bitwise_and(warped, warped, mask=ideal_mask)


@dataclasses.dataclass
class TactileCalibration:
    left_params: tuple[Any, Any, int, int]
    right_params: tuple[Any, Any, int, int]
    left_source: dict[str, Any]
    right_source: dict[str, Any]


def _compute_first_valid_live_tactile_calibration(camera, side_name: str, attempts: int):
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        frame_bgr, timestamp = _camera_get_bgr(camera)
        if frame_bgr is None:
            last_error = ValueError("camera returned no frame")
            time.sleep(0.03)
            continue
        try:
            params = _compute_tactile_calibration_params(frame_bgr)
            source = {
                "attempt": attempt,
                "timestamp": timestamp,
                "shape": list(frame_bgr.shape),
            }
            LOGGER.info(
                "%s tactile calibration succeeded from live frame attempt %d, shape=%s",
                side_name,
                attempt,
                frame_bgr.shape,
            )
            return params, source
        except ValueError as exc:
            last_error = exc
            if attempt == 1 or attempt % 20 == 0:
                LOGGER.warning("%s tactile calibration attempt %d failed: %s", side_name, attempt, exc)
            time.sleep(0.03)

    raise RuntimeError(
        f"Could not compute {side_name} tactile calibration from {attempts} live frames. "
        f"Last error: {last_error}"
    )


def _compute_live_tactile_calibration(left_camera, right_camera, attempts: int) -> TactileCalibration:
    left_params, left_source = _compute_first_valid_live_tactile_calibration(left_camera, "left", attempts)
    right_params, right_source = _compute_first_valid_live_tactile_calibration(right_camera, "right", attempts)
    return TactileCalibration(left_params, right_params, left_source, right_source)


def _save_tactile_calibration(session_dir: Path, calibration: TactileCalibration) -> None:
    data = {
        "left_H": calibration.left_params[0].tolist(),
        "left_mask": calibration.left_params[1].tolist(),
        "left_w": int(calibration.left_params[2]),
        "left_h": int(calibration.left_params[3]),
        "right_H": calibration.right_params[0].tolist(),
        "right_mask": calibration.right_params[1].tolist(),
        "right_w": int(calibration.right_params[2]),
        "right_h": int(calibration.right_params[3]),
        "source": {
            "left": calibration.left_source,
            "right": calibration.right_source,
        },
    }
    session_dir.mkdir(parents=True, exist_ok=True)
    calib_path = session_dir / "tactile_calib.json"
    with open(calib_path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    LOGGER.info("Saved live tactile calibration to %s", calib_path)


def _prepare_force(force: np.ndarray | None) -> np.ndarray:
    if force is None:
        return np.zeros((2,), dtype=np.float32)
    force_arr = np.asarray(force, dtype=np.float32).reshape(-1)
    if force_arr.shape[-1] < 2:
        force_arr = np.pad(force_arr, (0, 2 - force_arr.shape[-1]))
    return np.clip(force_arr[:2], 0.0, 10.0)


def _prepare_video_frame(
    image: np.ndarray | None, frame_shape: tuple[int, int], keep_aspect_ratio: bool = False
) -> np.ndarray | None:
    if image is None:
        return None

    frame = np.asarray(image)
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.ndim == 3 and frame.shape[2] == 1:
        frame = cv2.cvtColor(frame[:, :, 0], cv2.COLOR_GRAY2BGR)
    elif frame.ndim == 3 and frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    elif frame.ndim != 3 or frame.shape[2] != 3:
        LOGGER.warning("Skipping video frame with unsupported shape: %s", frame.shape)
        return None

    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)

    if (frame.shape[1], frame.shape[0]) != frame_shape:
        if keep_aspect_ratio:
            # 等比例缩放并加黑边 (Letterbox)，保持宽高比不被拉伸压缩
            target_w, target_h = frame_shape
            cur_h, cur_w = frame.shape[:2]
            scale = min(target_w / cur_w, target_h / cur_h)
            new_w = max(1, int(round(cur_w * scale)))
            new_h = max(1, int(round(cur_h * scale)))
            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            pad_h = (target_h - new_h) // 2
            pad_w = (target_w - new_w) // 2
            padded = np.zeros((target_h, target_w, 3), dtype=frame.dtype)
            padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
            frame = padded
        else:
            # 原本的暴力缩放
            frame = cv2.resize(frame, frame_shape, interpolation=cv2.INTER_AREA)

    return np.ascontiguousarray(frame)


def _concat_realsense_wrist_frame(
    realsense_img: np.ndarray | None,
    wrist_img: np.ndarray | None,
    frame_shape: tuple[int, int],
    *,
    highlight_wrist: bool = False,
) -> np.ndarray | None:
    # RealSense 因为是长方形，为了录像不被压扁，设置 keep_aspect_ratio=True
    realsense_frame = _prepare_video_frame(realsense_img, frame_shape, keep_aspect_ratio=True)
    # Wrist 是正方形，直接缩放即可
    wrist_frame = _prepare_video_frame(wrist_img, frame_shape, keep_aspect_ratio=False)

    if realsense_frame is None and wrist_frame is None:
        return None
    if realsense_frame is None:
        realsense_frame = np.zeros_like(wrist_frame)
    if wrist_frame is None:
        wrist_frame = np.zeros_like(realsense_frame)

    combined = np.ascontiguousarray(np.concatenate([realsense_frame, wrist_frame], axis=1))
    if highlight_wrist:
        x0 = frame_shape[0]
        cv2.rectangle(
            combined,
            (x0, 0),
            (combined.shape[1] - 1, combined.shape[0] - 1),
            (0, 0, 255),
            MODEL_INPUT_HIGHLIGHT_THICKNESS,
        )
    return combined


def _concat_synchronized_recording_frame(
    realsense_img: np.ndarray | None,
    realsense_third_img: np.ndarray | None,
    wrist_img: np.ndarray | None,
    left_tactile_img: np.ndarray | None,
    right_tactile_img: np.ndarray | None,
    frame_shape: tuple[int, int],
    *,
    include_realsense: bool,
    include_realsense_third: bool,
    include_tactile: bool,
    highlight_wrist: bool = False,
) -> np.ndarray | None:
    frames: list[np.ndarray | None] = []
    highlight_index: int | None = None

    if include_realsense:
        frames.append(_prepare_video_frame(realsense_img, frame_shape, keep_aspect_ratio=True))
    if include_realsense_third:
        frames.append(_prepare_video_frame(realsense_third_img, frame_shape, keep_aspect_ratio=True))

    highlight_index = len(frames)
    frames.append(_prepare_video_frame(wrist_img, frame_shape, keep_aspect_ratio=False))

    if include_tactile:
        frames.append(_prepare_video_frame(left_tactile_img, frame_shape, keep_aspect_ratio=True))
        frames.append(_prepare_video_frame(right_tactile_img, frame_shape, keep_aspect_ratio=True))

    reference_frame = next((frame for frame in frames if frame is not None), None)
    if reference_frame is None:
        return None

    filled_frames = [frame if frame is not None else np.zeros_like(reference_frame) for frame in frames]
    combined = np.ascontiguousarray(np.concatenate(filled_frames, axis=1))
    if highlight_wrist and highlight_index is not None:
        x0 = frame_shape[0] * highlight_index
        x1 = x0 + frame_shape[0] - 1
        cv2.rectangle(
            combined,
            (x0, 0),
            (x1, combined.shape[0] - 1),
            (0, 0, 255),
            MODEL_INPUT_HIGHLIGHT_THICKNESS,
        )
    return combined


def _highlight_video_frame(frame: np.ndarray) -> np.ndarray:
    highlighted = np.ascontiguousarray(frame.copy())
    cv2.rectangle(
        highlighted,
        (0, 0),
        (highlighted.shape[1] - 1, highlighted.shape[0] - 1),
        (0, 0, 255),
        MODEL_INPUT_HIGHLIGHT_THICKNESS,
    )
    return highlighted


def _make_policy_observation(latest_frame: dict[str, Any], prompt: str, use_tactile: bool) -> dict[str, Any]:
    if latest_frame["wrist_img"] is None:
        raise ValueError("Latest wrist image is missing")
    if use_tactile and latest_frame["left_tactile_img"] is None:
        raise ValueError("Latest left tactile image is missing")
    if use_tactile and latest_frame["right_tactile_img"] is None:
        raise ValueError("Latest right tactile image is missing")

    wrist_image = _prepare_camera_image(latest_frame["wrist_img"])
    left_tactile = _prepare_tactile_image(latest_frame["left_tactile_img"], use_tactile)
    right_tactile = _prepare_tactile_image(latest_frame["right_tactile_img"], use_tactile)
    state = encode_state_to_handcap_state(latest_frame["eepose"], float(latest_frame["gripper_width"]))
    force = _prepare_force(latest_frame.get("force"))

    return {
        "observation/wrist_image": wrist_image,
        "observation/left_tactile": left_tactile,
        "observation/right_tactile": right_tactile,
        "observation/state": state,
        "observation/force": force,
        # Keep LeRobot-style names too, in case this script is used with a server
        # configuration that includes HandcapRepackTransform.
        "observation.images.wrist": wrist_image,
        "observation.tactiles.left": left_tactile,
        "observation.tactiles.right": right_tactile,
        "observation.state": state,
        "observation.force": force,
        "prompt": prompt,
    }


@dataclasses.dataclass
class TrajectoryCostConfig:
    monitor_hz: float = 50.0
    w_e: float = 0.30
    w_m: float = 0.55
    w_c: float = 0.15
    t_ref: float = 10.0
    d_min: float = 0.02
    v_min: float = 0.005
    v_ref: float = 0.05
    a_ref: float = 0.30
    j_ref: float = 1.00
    a_max: float = 0.30
    f_ref: float = 10.0
    f_max: float = 30.0
    tau_ref: float = 5.0
    lambda_l: float = 1.0
    lambda_s: float = 1.0
    lambda_v: float = 1.0
    lambda_a: float = 1.0
    lambda_j: float = 1.0
    lambda_p: float = 1.0
    lambda_q: float = 1.0
    lambda_f: float = 1.0
    lambda_tau: float = 1.0


def _finite_float(value: Any, default: float | None = None) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _float_list(value: Any) -> list[float]:
    if value is None:
        return []
    try:
        return [float(x) for x in value]
    except (TypeError, ValueError):
        return []


def _vector(value: Any, min_size: int) -> np.ndarray | None:
    arr = np.asarray(_float_list(value), dtype=np.float64).reshape(-1)
    if arr.size < min_size or not np.all(np.isfinite(arr)):
        return None
    return arr


def _first_vector(min_size: int, *values: Any) -> np.ndarray | None:
    for value in values:
        vec = _vector(value, min_size)
        if vec is not None:
            return vec
    return None


def _rms_norm(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        norms = np.abs(arr)
    else:
        norms = np.linalg.norm(arr, axis=1)
    return float(np.sqrt(np.mean(np.square(norms)))) if norms.size else 0.0


def _p95_norm(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        norms = np.abs(arr)
    else:
        norms = np.linalg.norm(arr, axis=1)
    return float(np.percentile(norms, 95)) if norms.size else 0.0


def _quality_band(cost: float) -> str:
    if cost < 0.5:
        return "good"
    if cost < 1.0:
        return "acceptable"
    if cost < 2.0:
        return "fair"
    return "poor"


def _select_time(rows: list[dict[str, Any]]) -> tuple[np.ndarray, str]:
    for key in ("robot_time", "wall_time"):
        t = np.asarray([_finite_float(row.get(key), np.nan) for row in rows], dtype=np.float64)
        if np.all(np.isfinite(t)) and t[-1] > t[0]:
            return t, key
    t = np.arange(len(rows), dtype=np.float64)
    return t, "sample_index"


def compute_trajectory_cost(
    samples: list[dict[str, Any]],
    config: TrajectoryCostConfig,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for sample in samples:
        pose = _vector(sample.get("tcp_pose"), 7)
        vel = _vector(sample.get("tcp_vel"), 3)
        wall_time = _finite_float(sample.get("wall_time"))
        if pose is None or wall_time is None:
            continue
        rows.append(
            {
                "wall_time": wall_time,
                "robot_time": _finite_float(sample.get("robot_time")),
                "p": pose[:3],
                "v": vel[:3] if vel is not None else None,
                "dq": _first_vector(1, sample.get("dq"), sample.get("dtheta")),
                "dq_max": _vector(sample.get("dq_max"), 1),
                "f": _first_vector(
                    3,
                    sample.get("ext_wrench_in_world"),
                    sample.get("ext_wrench_in_tcp"),
                    sample.get("ft_sensor_raw"),
                ),
                "tau": _vector(sample.get("tau_ext"), 1),
                "target_tcp": _vector(sample.get("target_tcp"), 7),
            }
        )

    if len(rows) < 3:
        return {
            "valid": False,
            "reason": "Need at least 3 valid robot state samples with tcp_pose.",
            "num_samples": len(samples),
            "num_valid_samples": len(rows),
            "config": dataclasses.asdict(config),
        }

    t, time_source = _select_time(rows)
    keep = np.concatenate([[True], np.diff(t) > 1e-6])
    if keep.sum() < 3:
        return {
            "valid": False,
            "reason": "Robot state timestamps are not increasing.",
            "num_samples": len(samples),
            "num_valid_samples": int(keep.sum()),
            "time_source": time_source,
            "config": dataclasses.asdict(config),
        }

    t = t[keep]
    kept_rows = [row for row, flag in zip(rows, keep) if flag]
    p = np.vstack([row["p"] for row in kept_rows])
    dt = np.diff(t)

    v_rows = [row["v"] for row in kept_rows]
    if all(v is not None for v in v_rows):
        v = np.vstack(v_rows)
        velocity_source = "tcp_vel"
    else:
        segment_v = np.diff(p, axis=0) / dt[:, None]
        v = np.zeros_like(p)
        v[0] = segment_v[0]
        v[-1] = segment_v[-1]
        if len(p) > 2:
            v[1:-1] = 0.5 * (segment_v[:-1] + segment_v[1:])
        velocity_source = "pose_diff"
    speed = np.linalg.norm(v, axis=1)

    T = float(t[-1] - t[0])
    L = float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))
    D = float(np.linalg.norm(p[-1] - p[0]))
    effective_T = max(T, 1e-9)
    path_length_rate = L / effective_T
    displacement_rate = D / effective_T
    path_excess_distance = max(L - max(D, config.d_min), 0.0)
    path_excess_rate = path_excess_distance / effective_T
    path_rate_ref = max(config.v_ref, 1e-9)
    path_ratio = L / max(D, config.d_min)
    path_excess = max(path_ratio - 1.0, 0.0)
    r_stop = float(np.mean(speed < config.v_min))

    a = np.diff(v, axis=0) / dt[:, None]
    j = np.diff(a, axis=0) / dt[1:, None] if len(a) > 1 else np.empty((0, 3), dtype=np.float64)
    mu_v = float(np.mean(speed))
    sigma_v = float(np.std(speed))
    a_rms = _rms_norm(a)
    j_rms = _rms_norm(j)
    a_p95 = _p95_norm(a)

    dq_ratio_max = 0.0
    dq_available = False
    for row in kept_rows:
        dq = row["dq"]
        dq_max = row["dq_max"]
        if dq is None or dq_max is None:
            continue
        width = min(dq.size, dq_max.size)
        if width == 0:
            continue
        denom = np.maximum(np.abs(dq_max[:width]), 1e-9)
        dq_ratio_max = max(dq_ratio_max, float(np.max(np.abs(dq[:width]) / denom)))
        dq_available = True

    f_rows = [row["f"][:3] for row in kept_rows if row["f"] is not None]
    f = np.vstack(f_rows) if f_rows else np.empty((0, 3), dtype=np.float64)
    tau_rows = [row["tau"] for row in kept_rows if row["tau"] is not None]
    tau = np.vstack(tau_rows) if tau_rows else np.empty((0, 0), dtype=np.float64)
    f_rms = _rms_norm(f)
    f_p95 = _p95_norm(f)
    tau_rms = _rms_norm(tau)

    target_errors = []
    for row in kept_rows:
        target_tcp = row["target_tcp"]
        if target_tcp is not None:
            target_errors.append(float(np.linalg.norm(row["p"] - target_tcp[:3])))
    target_pos_error_rms = (
        float(np.sqrt(np.mean(np.square(target_errors)))) if target_errors else None
    )

    eps = 1e-9
    path_length_rate_cost = path_length_rate / max(path_rate_ref, eps)
    path_excess_rate_cost = path_excess_rate / max(path_rate_ref, eps)
    # Efficiency cost is normalized by elapsed time. Using whole-trajectory
    # duration or path length directly makes early safety-boundary exits look
    # artificially cheap compared with successful, full-length rollouts.
    J_e = (
        config.lambda_l * path_length_rate_cost
        + path_excess_rate_cost
        + config.lambda_s * r_stop
    )
    J_m = (
        config.lambda_v * sigma_v / (mu_v + eps)
        + config.lambda_a * a_rms / max(config.a_ref, eps)
        + config.lambda_j * j_rms / max(config.j_ref, eps)
        + config.lambda_p * a_p95 / max(config.a_max, eps)
        + config.lambda_q * dq_ratio_max
    )
    J_c = (
        f_rms / max(config.f_ref, eps)
        + config.lambda_f * f_p95 / max(config.f_max, eps)
        + config.lambda_tau * tau_rms / max(config.tau_ref, eps)
    )
    J = config.w_e * J_e + config.w_m * J_m + config.w_c * J_c

    return {
        "valid": True,
        "J": float(J),
        "J_e": float(J_e),
        "J_m": float(J_m),
        "J_c": float(J_c),
        "quality_band": _quality_band(float(J)),
        "time_source": time_source,
        "velocity_source": velocity_source,
        "num_samples": len(samples),
        "num_valid_samples": len(kept_rows),
        "sample_hz_est": float((len(t) - 1) / T) if T > 0 else 0.0,
        "components": {
            "T": T,
            "L": L,
            "D": D,
            "path_length_rate": float(path_length_rate),
            "displacement_rate": float(displacement_rate),
            "path_excess_distance": float(path_excess_distance),
            "path_excess_rate": float(path_excess_rate),
            "path_rate_ref": float(path_rate_ref),
            "path_length_rate_cost": float(path_length_rate_cost),
            "path_excess_rate_cost": float(path_excess_rate_cost),
            "path_ratio": float(path_ratio),
            "path_excess": float(path_excess),
            "legacy_time_cost": float(T / max(config.t_ref, eps)),
            "r_stop": r_stop,
            "mu_v": mu_v,
            "sigma_v": sigma_v,
            "a_rms": a_rms,
            "j_rms": j_rms,
            "a_p95": a_p95,
            "dq_ratio_max": dq_ratio_max,
            "dq_available": dq_available,
            "f_rms": f_rms,
            "f_p95": f_p95,
            "tau_rms": tau_rms,
            "target_pos_error_rms": target_pos_error_rms,
        },
        "config": dataclasses.asdict(config),
    }


class TrajectoryMetricMonitor(threading.Thread):
    def __init__(self, env: "FlexivRealEnv", config: TrajectoryCostConfig):
        super().__init__(daemon=True)
        self.env = env
        self.config = config
        self.running = True
        self.lock = threading.Lock()
        self.samples: list[dict[str, Any]] = []
        self.error: Exception | None = None

    def run(self) -> None:
        period = 1.0 / max(self.config.monitor_hz, 1e-6)
        while self.running:
            start = time.time()
            try:
                sample = self.env.get_metric_snapshot()
                with self.lock:
                    self.samples.append(sample)
            except Exception as exc:  # pragma: no cover - hardware state path
                if self.error is None:
                    LOGGER.warning("Trajectory metric monitor failed to sample robot state: %s", exc)
                self.error = exc
            elapsed = time.time() - start
            sleep_time = period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def stop(self) -> None:
        self.running = False

    def snapshot(self) -> list[dict[str, Any]]:
        with self.lock:
            return list(self.samples)


class RuntimeControl:
    def __init__(self, control_file: Path | None):
        self.control_file = control_file
        self.event_file = control_file.parent / f"{control_file.name}.events" if control_file is not None else None
        self.start_requested = threading.Event()
        self.stop_requested = threading.Event()
        self.reset_requested = threading.Event()
        self._decision_queue: queue.Queue[str] = queue.Queue()
        self._running = True
        self._thread: threading.Thread | None = None
        self._offset = 0

        if self.control_file is None:
            self.start_requested.set()

    def start(self) -> None:
        if self.control_file is None:
            return

        self.control_file.parent.mkdir(parents=True, exist_ok=True)
        self.control_file.touch(exist_ok=True)
        if self.event_file is not None:
            self.event_file.touch(exist_ok=True)
        self._thread = threading.Thread(target=self._watch_control_file, daemon=True, name="handcap-runtime-control")
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def consume_reset(self) -> bool:
        if not self.reset_requested.is_set():
            return False
        self.reset_requested.clear()
        return True

    def consume_start(self) -> bool:
        if not self.start_requested.is_set():
            return False
        self.start_requested.clear()
        return True

    def consume_stop(self) -> bool:
        if not self.stop_requested.is_set():
            return False
        self.stop_requested.clear()
        return True

    def emit_event(self, name: str, payload: str | Path | None = None) -> None:
        if self.event_file is None:
            return
        line = name if payload is None else f"{name}\t{payload}"
        try:
            with open(self.event_file, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception as exc:  # pragma: no cover - control file is best-effort
            LOGGER.warning("Failed to write runtime control event %s: %s", name, exc)

    def wait_for_save_decision(self) -> str:
        if self.control_file is None:
            return "keep"
        while True:
            try:
                return self._decision_queue.get(timeout=0.1)
            except queue.Empty:
                if not self._running:
                    return "keep"

    def _watch_control_file(self) -> None:
        assert self.control_file is not None
        while self._running:
            try:
                size = self.control_file.stat().st_size
                if size < self._offset:
                    self._offset = 0
                if size > self._offset:
                    with open(self.control_file, "r", encoding="utf-8") as f:
                        f.seek(self._offset)
                        chunk = f.read()
                        self._offset = f.tell()
                    self._handle_commands(chunk.splitlines())
            except FileNotFoundError:
                pass
            except Exception as exc:  # pragma: no cover - control file is best-effort
                LOGGER.warning("Runtime control file watcher skipped a poll: %s", exc)
            time.sleep(0.05)

    def _handle_commands(self, commands: list[str]) -> None:
        for command in commands:
            command = command.strip().lower()
            if command == "start":
                if not self.start_requested.is_set():
                    LOGGER.info("\033[92mReceived SPACE/start command. Recording and inference will start.\033[0m")
                self.start_requested.set()
            elif command == "stop":
                LOGGER.info("\033[93mReceived SPACE/stop command. Current episode will finish and flush.\033[0m")
                self.stop_requested.set()
            elif command == "reset":
                LOGGER.info("\033[93mReceived r/reset command. Reset will run at the next safe point.\033[0m")
                self.reset_requested.set()
            elif command in {"keep", "discard"}:
                self._decision_queue.put(command)


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000
    obs_horizon: int = 2
    ctrl_freq: float = 20.0
    steps_per_inference: int = 4
    action_latency: float = 0.0
    task_name: str = "handcap_flexiv_mvs"
    prompt: str = ""
    robot_ip: str = ""
    local_ip: str = dataclasses.field(default_factory=lambda: os.environ.get("FLEXIV_LOCAL_IP", ""))
    init_qpos: tuple[float, ...] = (-0.0, -0.698, -0.0, 1.571, -0.0, 0.698, -0.0)
    camera_config_path: Path = dataclasses.field(default_factory=_default_camera_config_path)
    control_file: Path | None = None
    record_root: str = "real_robot_inference_recording"
    use_tactile: bool = True
    tactile_calibration: bool = True
    tactile_calibration_attempts: int = 120
    force_predict: bool = False
    force_guide: bool = False
    infer_force_control: bool = False
    dry_run: bool = False
    save_states_json: bool = True
    record_hz: float = 20.0
    metric_monitor_hz: float = 50.0
    metric_t_ref: float = 10.0
    metric_d_min: float = 0.02
    metric_v_min: float = 0.005
    metric_v_ref: float = 0.05
    metric_a_ref: float = 0.30
    metric_j_ref: float = 1.00
    metric_a_max: float = 0.30
    metric_f_ref: float = 10.0
    metric_f_max: float = 30.0
    metric_tau_ref: float = 5.0
    metric_lambda_l: float = 1.0
    metric_lambda_s: float = 1.0
    metric_lambda_v: float = 1.0
    metric_lambda_a: float = 1.0
    metric_lambda_j: float = 1.0
    metric_lambda_p: float = 1.0
    metric_lambda_q: float = 1.0
    metric_lambda_f: float = 1.0
    metric_lambda_tau: float = 1.0


def _make_trajectory_cost_config(args: Args) -> TrajectoryCostConfig:
    return TrajectoryCostConfig(
        monitor_hz=args.metric_monitor_hz,
        t_ref=args.metric_t_ref,
        d_min=args.metric_d_min,
        v_min=args.metric_v_min,
        v_ref=args.metric_v_ref,
        a_ref=args.metric_a_ref,
        j_ref=args.metric_j_ref,
        a_max=args.metric_a_max,
        f_ref=args.metric_f_ref,
        f_max=args.metric_f_max,
        tau_ref=args.metric_tau_ref,
        lambda_l=args.metric_lambda_l,
        lambda_s=args.metric_lambda_s,
        lambda_v=args.metric_lambda_v,
        lambda_a=args.metric_lambda_a,
        lambda_j=args.metric_lambda_j,
        lambda_p=args.metric_lambda_p,
        lambda_q=args.metric_lambda_q,
        lambda_f=args.metric_lambda_f,
        lambda_tau=args.metric_lambda_tau,
    )


class Recorder:
    def __init__(
        self,
        output_dir: Path,
        episode_index: int,
        video_fps: float,
        wrist_shape: tuple[int, int],
        realsense_shape: tuple[int, int] | None,
        realsense_third_shape: tuple[int, int] | None,
        left_shape: tuple[int, int] | None,
        right_shape: tuple[int, int] | None,
        *,
        realsense_source: Any | None,
        realsense_third_source: Any | None,
        save_states_json: bool,
    ) -> None:
        self.output_dir = output_dir
        self.episode_index = int(episode_index)
        self.realsense_source = realsense_source
        self.realsense_third_source = realsense_third_source
        self.save_states_json = save_states_json
        self.states_data: list[dict[str, Any]] = []
        self._closed = False
        self._writer_running = True
        self._frame_queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=2)
        self.metric_monitor: TrajectoryMetricMonitor | None = None
        self.metric_config: TrajectoryCostConfig | None = None

        output_dir.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        self.wrist_shape = wrist_shape
        self.realsense_shape = realsense_shape
        self.realsense_third_shape = realsense_third_shape
        self.left_shape = left_shape
        self.right_shape = right_shape
        self._video_paths: dict[str, Path] = {}
        self._video_frame_counts: dict[str, int] = {"wrist": 0}
        self.video_fps = float(video_fps)
        episode_prefix = f"episode_{self.episode_index:03d}"
        self.record_tactile_in_main_video = left_shape is not None and right_shape is not None
        main_views = 1
        if realsense_shape is not None:
            main_views += 1
        if realsense_third_shape is not None:
            main_views += 1
        if self.record_tactile_in_main_video:
            main_views += 2

        if realsense_shape is not None and self.record_tactile_in_main_video:
            wrist_video_name = f"{episode_prefix}_view1_realsense_wrist_tactile.raw.avi"
        elif realsense_shape is not None:
            wrist_video_name = f"{episode_prefix}_view1_realsense_wrist.raw.avi"
        elif self.record_tactile_in_main_video:
            wrist_video_name = f"{episode_prefix}_view1_wrist_tactile.raw.avi"
        else:
            wrist_video_name = f"{episode_prefix}_view1_wrist.raw.avi"
        wrist_video_shape = (wrist_shape[0] * main_views, wrist_shape[1])
        self._video_paths["wrist"] = output_dir / wrist_video_name
        self.wrist_video = cv2.VideoWriter(str(self._video_paths["wrist"]), fourcc, video_fps, wrist_video_shape)
        if not self.wrist_video.isOpened():
            LOGGER.warning("Failed to open wrist video writer at %s; only state logs will be saved.", self._video_paths["wrist"])
            self.wrist_video = None
        else:
            LOGGER.info(
                "Recording synchronized main video (%d views: %s) to %s",
                main_views,
                "realsense|wrist|left_tactile|right_tactile"
                if realsense_shape is not None and self.record_tactile_in_main_video
                else "realsense|wrist"
                if realsense_shape is not None
                else "wrist|left_tactile|right_tactile"
                if self.record_tactile_in_main_video
                else "wrist",
                self._video_paths["wrist"],
            )
        self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True, name="handcap-video-writer")
        self._writer_thread.start()

    def attach_metric_monitor(
        self,
        metric_monitor: TrajectoryMetricMonitor,
        metric_config: TrajectoryCostConfig,
    ) -> None:
        self.metric_monitor = metric_monitor
        self.metric_config = metric_config

    def record_frame(self, latest_frame: dict[str, Any], *, model_input: bool = False) -> None:
        if self._closed:
            return

        packet = {
            "wrist_img": latest_frame.get("wrist_img"),
            "realsense_img": self.realsense_source.get_frame(copy_frame=False)
            if self.realsense_source is not None
            else None,
            "realsense_third_img": self.realsense_third_source.get_frame(copy_frame=False)
            if self.realsense_third_source is not None
            else None,
            "left_tactile_img": latest_frame.get("left_tactile_img"),
            "right_tactile_img": latest_frame.get("right_tactile_img"),
            "model_input": model_input,
        }
        try:
            self._frame_queue.put_nowait(packet)
        except queue.Full:
            try:
                self._frame_queue.get_nowait()
                self._frame_queue.task_done()
            except queue.Empty:
                pass
            try:
                self._frame_queue.put_nowait(packet)
            except queue.Full:
                pass

    def _writer_loop(self) -> None:
        while self._writer_running or not self._frame_queue.empty():
            try:
                latest_frame = self._frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self._write_frame(latest_frame)
            except Exception as exc:  # pragma: no cover - recording should never stop control
                LOGGER.warning("Failed to write recording frame: %s", exc)
            finally:
                self._frame_queue.task_done()

    def _write_frame(self, latest_frame: dict[str, Any]) -> None:
        if self.wrist_video is not None:
            wrist_frame = _concat_synchronized_recording_frame(
                latest_frame.get("realsense_img"),
                latest_frame.get("realsense_third_img"),
                latest_frame["wrist_img"],
                latest_frame.get("left_tactile_img"),
                latest_frame.get("right_tactile_img"),
                self.wrist_shape,
                include_realsense=self.realsense_shape is not None,
                include_realsense_third=self.realsense_third_shape is not None,
                include_tactile=self.record_tactile_in_main_video,
                highlight_wrist=bool(latest_frame.get("model_input")),
            )
            if wrist_frame is not None:
                self.wrist_video.write(wrist_frame)
                self._video_frame_counts["wrist"] += 1

    def _run_ffmpeg_transcode(self, source_path: Path, mp4_path: Path, codec_args: list[str]) -> subprocess.CompletedProcess:
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-fflags",
            "+genpts",
            "-r",
            f"{self.video_fps:g}",
            "-i",
            str(source_path),
            "-an",
            *codec_args,
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(mp4_path),
        ]
        return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)

    def _opencv_transcode_playable_mp4(self, source_path: Path, mp4_path: Path) -> Path | None:
        cap = cv2.VideoCapture(str(source_path))
        if not cap.isOpened():
            LOGGER.warning("OpenCV could not read raw video for MP4 fallback: %s", source_path)
            return None

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or self.video_fps)
        if width <= 0 or height <= 0:
            cap.release()
            LOGGER.warning("OpenCV fallback got invalid video size for %s", source_path)
            return None

        writer = cv2.VideoWriter(
            str(mp4_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps if fps > 0 else self.video_fps,
            (width, height),
        )
        if not writer.isOpened():
            cap.release()
            LOGGER.warning("OpenCV could not open MP4 writer for fallback: %s", mp4_path)
            return None

        frame_count = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
            frame_count += 1

        cap.release()
        writer.release()

        if frame_count == 0 or not mp4_path.exists() or mp4_path.stat().st_size == 0:
            try:
                mp4_path.unlink(missing_ok=True)
            except OSError:
                pass
            LOGGER.warning("OpenCV MP4 fallback produced no usable video for %s", source_path)
            return None

        LOGGER.info(
            "Playable MP4 saved via OpenCV fallback: frames=%d size=%.1fKB path=%s",
            frame_count,
            mp4_path.stat().st_size / 1024.0,
            mp4_path,
        )
        return mp4_path

    def _transcode_playable_mp4(self, source_path: Path) -> Path | None:
        if not source_path.exists() or source_path.stat().st_size == 0:
            return None
        mp4_path = source_path.with_suffix("").with_suffix(".mp4") if source_path.name.endswith(".raw.avi") else source_path.with_suffix(".mp4")
        if shutil.which("ffmpeg") is None:
            LOGGER.warning("ffmpeg is not available; using OpenCV MP4 fallback for %s", source_path)
            return self._opencv_transcode_playable_mp4(source_path, mp4_path)

        result = self._run_ffmpeg_transcode(
            source_path,
            mp4_path,
            ["-c:v", "libx264", "-crf", "18", "-preset", "veryfast"],
        )
        if result.returncode != 0:
            LOGGER.warning("libx264 MP4 transcode failed for %s; retrying with MPEG-4. Error: %s", source_path, result.stderr.strip())
            result = self._run_ffmpeg_transcode(
                source_path,
                mp4_path,
                ["-c:v", "mpeg4", "-q:v", "3"],
            )
        if result.returncode != 0:
            LOGGER.warning("ffmpeg failed to transcode %s to MP4; using OpenCV fallback. Error: %s", source_path, result.stderr.strip())
            return self._opencv_transcode_playable_mp4(source_path, mp4_path)
        if not mp4_path.exists() or mp4_path.stat().st_size == 0:
            LOGGER.warning("ffmpeg produced no MP4 for %s; using OpenCV fallback.", source_path)
            return self._opencv_transcode_playable_mp4(source_path, mp4_path)
        LOGGER.info("Playable MP4 saved on server: %.1fKB path=%s", mp4_path.stat().st_size / 1024.0, mp4_path)
        return mp4_path

    def append_state(
        self,
        *,
        step: int,
        inference_latency: float,
        latest_frame: dict[str, Any],
        action_chunk: np.ndarray,
        client_timing: dict[str, Any] | None,
        server_timing: dict[str, Any] | None,
        policy_timing: dict[str, Any] | None,
    ) -> None:
        state_entry = {
            "step": step,
            "inference_latency": float(inference_latency),
            "robot0_eef_pos_x": float(latest_frame["eepose"][0]),
            "robot0_eef_pos_y": float(latest_frame["eepose"][1]),
            "robot0_eef_pos_z": float(latest_frame["eepose"][2]),
            "robot0_eef_rot_x": float(latest_frame["eepose"][3]),
            "robot0_eef_rot_y": float(latest_frame["eepose"][4]),
            "robot0_eef_rot_z": float(latest_frame["eepose"][5]),
            "gripper_width": float(latest_frame["gripper_width"]),
            "action_chunk_size": int(action_chunk.shape[0]),
        }
        obs_timing = latest_frame.get("timing")
        if isinstance(obs_timing, dict):
            for key, value in obs_timing.items():
                state_key = key if key.startswith("obs_") else f"obs_{key}"
                state_entry[state_key] = float(value)
        if "force" in latest_frame:
            force = _prepare_force(latest_frame.get("force"))
            state_entry["force_left"] = float(force[0])
            state_entry["force_right"] = float(force[1])
        if action_chunk.shape[-1] >= 12:
            state_entry["pred_force_left"] = float(action_chunk[0, 10])
            state_entry["pred_force_right"] = float(action_chunk[0, 11])
        if client_timing is not None:
            for key, value in client_timing.items():
                state_entry[f"client_{key}"] = float(value)
        if server_timing is not None:
            for key, value in server_timing.items():
                state_entry[f"server_{key}"] = float(value)
        if policy_timing is not None:
            for key, value in policy_timing.items():
                state_entry[f"policy_{key}"] = float(value)
        self.states_data.append(state_entry)

    def flush(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._writer_running = False
        self._writer_thread.join(timeout=10.0)
        if self._writer_thread.is_alive():
            LOGGER.warning("Video writer thread did not finish before timeout; releasing writers anyway.")

        if self.wrist_video is not None:
            self.wrist_video.release()

        for key, path in self._video_paths.items():
            if path.exists():
                LOGGER.info(
                    "Recording saved: %s frames=%d size=%.1fKB path=%s",
                    key,
                    self._video_frame_counts.get(key, 0),
                    path.stat().st_size / 1024.0,
                    path,
                )
                mp4_path = self._transcode_playable_mp4(path)
                if mp4_path is not None and path.suffix == ".avi":
                    try:
                        path.unlink()
                        LOGGER.info("Removed intermediate raw AVI: %s", path)
                    except OSError as exc:
                        LOGGER.warning("Failed to remove intermediate raw AVI %s: %s", path, exc)
            else:
                LOGGER.warning("Recording file was not created: %s path=%s", key, path)
        if self.states_data:
            if self.save_states_json:
                with open(self.output_dir / "states.json", "w", encoding="utf-8") as f:
                    json.dump(self.states_data, f)
            try:
                import polars as pl

                pl.DataFrame(self.states_data).write_parquet(str(self.output_dir / "states.parquet"))
            except ModuleNotFoundError:
                try:
                    import pandas as pd

                    pd.DataFrame(self.states_data).to_parquet(str(self.output_dir / "states.parquet"))
                except Exception as exc:  # pragma: no cover - environment-specific parquet backend failure
                    LOGGER.warning("Failed to write states.parquet with pandas fallback: %s", exc)
            except Exception as exc:  # pragma: no cover - environment-specific parquet writer failure
                LOGGER.warning("Failed to write states.parquet with polars: %s", exc)

        if self.metric_monitor is not None and self.metric_config is not None:
            metric_samples = self.metric_monitor.snapshot()
            metric_summary = compute_trajectory_cost(metric_samples, self.metric_config)
            with open(self.output_dir / "trajectory_cost.json", "w", encoding="utf-8") as f:
                json.dump(metric_summary, f, indent=2)
            if self.save_states_json:
                with open(self.output_dir / "trajectory_metric_samples.json", "w", encoding="utf-8") as f:
                    json.dump(metric_samples, f)
            if metric_samples:
                try:
                    import polars as pl

                    pl.DataFrame(metric_samples).write_parquet(str(self.output_dir / "trajectory_metric_samples.parquet"))
                except ModuleNotFoundError:
                    try:
                        import pandas as pd

                        pd.DataFrame(metric_samples).to_parquet(
                            str(self.output_dir / "trajectory_metric_samples.parquet")
                        )
                    except Exception as exc:  # pragma: no cover - environment-specific parquet backend failure
                        LOGGER.warning("Failed to write trajectory_metric_samples.parquet with pandas fallback: %s", exc)
                except Exception as exc:  # pragma: no cover - environment-specific parquet writer failure
                    LOGGER.warning("Failed to write trajectory_metric_samples.parquet with polars: %s", exc)
            LOGGER.info(
                "Trajectory cost saved to %s: J=%s, J_e=%s, J_m=%s, J_c=%s",
                self.output_dir / "trajectory_cost.json",
                metric_summary.get("J"),
                metric_summary.get("J_e"),
                metric_summary.get("J_m"),
                metric_summary.get("J_c"),
            )

        if hasattr(os, "sync"):
            os.sync()


def _signal_handler(sig, frame) -> None:
    del sig, frame
    global _SHUTTING_DOWN
    if _SHUTTING_DOWN:
        return
    _SHUTTING_DOWN = True
    signal_module.signal(signal_module.SIGINT, signal_module.SIG_IGN)
    signal_module.signal(signal_module.SIGTERM, signal_module.SIG_IGN)
    LOGGER.info("Detected interrupt signal; exiting after normal cleanup.")
    raise KeyboardInterrupt


class LatestCameraThread(threading.Thread):
    def __init__(self, camera, name: str):
        super().__init__(daemon=True, name=f"handcap-{name}-capture")
        self.camera = camera
        self.name_for_log = name
        self.running = True
        self.lock = threading.Lock()
        self.frame: np.ndarray | None = None
        self.timestamp: float | None = None
        self.error: Exception | None = None

    def run(self) -> None:
        try:
            while self.running:
                ok, frame, timestamp = self.camera.read()
                if not ok or frame is None:
                    continue
                with self.lock:
                    self.frame = frame.copy()
                    self.timestamp = float(timestamp) if timestamp is not None else time.time()
        except Exception as exc:  # pragma: no cover - optional recording camera failure
            self.error = exc
            self.running = False
            LOGGER.warning("Optional %s recording camera stopped: %s", self.name_for_log, exc)

    def get_frame(self, *, copy_frame: bool = True) -> np.ndarray | None:
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy() if copy_frame else self.frame

    def stop(self) -> None:
        self.running = False


class ObservationThread(threading.Thread):
    def __init__(
        self,
        cam_wrist,
        cam_tactile_left,
        cam_tactile_right,
        env,
        maxlen: int = 2,
        tactile_calibration: TactileCalibration | None = None,
    ):
        super().__init__(daemon=True)
        self.cam_wrist = cam_wrist
        self.cam_tactile_left = cam_tactile_left
        self.cam_tactile_right = cam_tactile_right
        self.env = env
        self.maxlen = maxlen
        self.tactile_calibration = tactile_calibration
        self.running = True
        self.lock = threading.Lock()
        self.queue: list[dict[str, Any]] = []
        self._obs_id = 0
        self.error: Exception | None = None

    def run(self) -> None:
        try:
            while self.running:
                obs_start = time.monotonic()
                wrist_start = time.monotonic()
                _, wrist_img, cam_cap_time = self.cam_wrist.read()
                wrist_read_ms = (time.monotonic() - wrist_start) * 1000.0
                left_tactile_img, right_tactile_img = None, None
                left_tactile_read_ms = 0.0
                right_tactile_read_ms = 0.0
                if self.cam_tactile_left is not None:
                    tactile_start = time.monotonic()
                    left_tactile_img, _ = _camera_get_bgr(self.cam_tactile_left)
                    left_tactile_read_ms = (time.monotonic() - tactile_start) * 1000.0
                if self.cam_tactile_right is not None:
                    tactile_start = time.monotonic()
                    right_tactile_img, _ = _camera_get_bgr(self.cam_tactile_right)
                    right_tactile_read_ms = (time.monotonic() - tactile_start) * 1000.0

                robot_state_start = time.monotonic()
                eepose = self.env.get_ee_pose()
                gripper_width = self.env.get_gripper_width()
                force = self.env.get_force()
                robot_state_ms = (time.monotonic() - robot_state_start) * 1000.0

                preprocess_start = time.monotonic()
                if wrist_img is not None:
                    wrist_img = _prepare_wrist_like_collection(wrist_img)
                if self.tactile_calibration is not None:
                    if left_tactile_img is not None:
                        left_tactile_img = _apply_tactile_calibration(
                            left_tactile_img, self.tactile_calibration.left_params
                        )
                    if right_tactile_img is not None:
                        right_tactile_img = _apply_tactile_calibration(
                            right_tactile_img, self.tactile_calibration.right_params
                        )
                preprocess_ms = (time.monotonic() - preprocess_start) * 1000.0
                obs_total_ms = (time.monotonic() - obs_start) * 1000.0

                with self.lock:
                    obs_id = self._obs_id
                    self._obs_id += 1
                    self.queue.append(
                        {
                            "obs_id": obs_id,
                            "wrist_img": wrist_img.copy() if wrist_img is not None else None,
                            "left_tactile_img": left_tactile_img.copy() if left_tactile_img is not None else None,
                            "right_tactile_img": right_tactile_img.copy() if right_tactile_img is not None else None,
                            "cam_cap_time": cam_cap_time,
                            "eepose": np.asarray(eepose, dtype=np.float64),
                            "gripper_width": float(gripper_width),
                            "force": np.asarray(force, dtype=np.float32),
                            "timing": {
                                "obs_total_ms": obs_total_ms,
                                "camera_read_ms": wrist_read_ms + left_tactile_read_ms + right_tactile_read_ms,
                                "wrist_read_ms": wrist_read_ms,
                                "left_tactile_read_ms": left_tactile_read_ms,
                                "right_tactile_read_ms": right_tactile_read_ms,
                                "robot_state_ms": robot_state_ms,
                                "preprocess_ms": preprocess_ms,
                            },
                        }
                    )
                    self.queue = self.queue[-self.maxlen :]
        except Exception as exc:  # pragma: no cover - hardware thread failure path
            self.error = exc
            self.running = False

    def get_obs(self, n: int = 2) -> list[dict[str, Any]] | None:
        if self.error is not None:
            raise RuntimeError("Observation thread failed") from self.error
        with self.lock:
            if len(self.queue) < n:
                return None
            return list(self.queue[-n:])

    def stop(self) -> None:
        self.running = False

    def get_latest(self) -> dict[str, Any] | None:
        if self.error is not None:
            raise RuntimeError("Observation thread failed") from self.error
        with self.lock:
            if not self.queue:
                return None
            return dict(self.queue[-1])


class ObservationRecordingThread(threading.Thread):
    def __init__(self, obs_thread: ObservationThread, recorder: Recorder, record_hz: float):
        super().__init__(daemon=True, name="handcap-observation-recorder")
        self.obs_thread = obs_thread
        self.recorder = recorder
        self.record_hz = float(record_hz)
        self.running = True
        self._last_obs_id: int | None = None

    def run(self) -> None:
        period = 1.0 / max(self.record_hz, 1e-6)
        while self.running:
            start = time.time()
            try:
                latest_frame = self.obs_thread.get_latest()
                if latest_frame is not None:
                    obs_id = latest_frame.get("obs_id")
                    if obs_id != self._last_obs_id:
                        self.recorder.record_frame(latest_frame)
                        self._last_obs_id = obs_id
            except Exception as exc:  # pragma: no cover - recording should not affect inference
                LOGGER.warning("Observation recording thread skipped a frame: %s", exc)
            elapsed = time.time() - start
            sleep_time = period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def stop(self) -> None:
        self.running = False


class FlexivRealEnv:
    def __init__(
        self,
        init_qpos: list[float] | tuple[float, ...],
        *,
        robot_ip: str,
        local_ip: str,
        dry_run: bool,
    ) -> None:
        import flexivrdk

        self._flexivrdk = flexivrdk
        self._mode = flexivrdk.Mode
        self._dry_run = dry_run
        self._init_qpos = list(init_qpos) if init_qpos is not None else None
        self._arm_max_linear_vel = _required_env_float("FLEXIV_ARM_MAX_LINEAR_VEL")
        self._arm_max_angular_vel = _required_env_float("FLEXIV_ARM_MAX_ANGULAR_VEL")
        self._arm_max_linear_acc = _required_env_float("FLEXIV_ARM_MAX_LINEAR_ACC")
        self._arm_max_angular_acc = _required_env_float("FLEXIV_ARM_MAX_ANGULAR_ACC")
        self._gripper_move_velocity = _required_env_float("FLEXIV_GRIPPER_MOVE_VELOCITY")
        self._gripper_move_force = _required_env_float("FLEXIV_GRIPPER_MOVE_FORCE")
        self._enable_safety_clip = _required_env_bool("FLEXIV_ENABLE_SAFETY_CLIP")
        self._action_frame = os.environ.get("FLEXIV_ACTION_FRAME", "tcp").lower()
        if self._action_frame not in {"tcp", "flange"}:
            LOGGER.warning("Invalid FLEXIV_ACTION_FRAME=%r; using 'tcp'.", self._action_frame)
            self._action_frame = "tcp"

        robot_sn = os.environ.get("FLEXIV_ROBOT_SN")
        if not robot_sn:
            raise ValueError("Set FLEXIV_ROBOT_SN to your robot serial number.")
        gripper_name = os.environ.get("FLEXIV_GRIPPER_NAME", "Flexiv-GN01")
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.connect((robot_ip, 1))
            actual_local_ip = sock.getsockname()[0]
            sock.close()
        except Exception:
            actual_local_ip = local_ip

        LOGGER.info(
            "Initializing Flexiv Robot SN=%s, robot_ip=%s, local_ip=%s",
            robot_sn,
            robot_ip,
            actual_local_ip,
        )
        LOGGER.info(
            "Flexiv Cartesian EEF limits: action_frame=%s linear_vel=%.3f angular_vel=%.3f "
            "linear_acc=%.3f angular_acc=%.3f gripper_vel=%.3f gripper_force=%.3f safety_clip=%s",
            self._action_frame,
            self._arm_max_linear_vel,
            self._arm_max_angular_vel,
            self._arm_max_linear_acc,
            self._arm_max_angular_acc,
            self._gripper_move_velocity,
            self._gripper_move_force,
            self._enable_safety_clip,
        )
        self._robot = flexivrdk.Robot(robot_sn, [actual_local_ip])

        LOGGER.info("Initializing Gripper API.")
        self._gripper = flexivrdk.Gripper(self._robot)
        LOGGER.info("Initializing Model API.")
        self._model = flexivrdk.Model(self._robot)
        self._target_lock = threading.Lock()
        self._latest_target_tcp: list[float] | None = None
        self._latest_target_time: float | None = None
        try:
            self._dq_max = _float_list(getattr(self._robot.info(), "dq_max", None))
        except Exception:
            self._dq_max = []

        try:
            LOGGER.info("Enabling gripper [%s].", gripper_name)
            self._gripper.Enable(gripper_name)
        except Exception as exc:
            LOGGER.warning("Failed to enable gripper [%s]: %s", gripper_name, exc)

        if self._robot.fault():
            self._robot.ClearFault()
            time.sleep(2)

        self._robot.Enable()
        while not self._robot.operational():
            time.sleep(1)

        LOGGER.info("Switching to NRT_CARTESIAN_MOTION_FORCE mode.")
        self._robot.SwitchMode(self._mode.NRT_CARTESIAN_MOTION_FORCE)
        self._robot.SetForceControlAxis([False, False, False, False, False, False])

        if not self._dry_run:
            max_width = self._gripper.params().max_width
            self._gripper.Move(max_width, self._gripper_move_velocity, self._gripper_move_force)
            time.sleep(1)

    def reset(self) -> None:
        if self._init_qpos is None:
            return

        LOGGER.info("Switching to NRT_JOINT_POSITION for reset.")
        self._robot.SwitchMode(self._mode.NRT_JOINT_POSITION)
        dof = len(self._init_qpos)
        if not self._dry_run:
            LOGGER.info("\033[93mResetting robot to initial joint positions: %s\033[0m", self._init_qpos)
            self._robot.SendJointPosition(self._init_qpos, [0.0] * dof, [0.1] * dof, [0.1] * dof)
            max_width = self._gripper.params().max_width
            self._gripper.Move(max_width, self._gripper_move_velocity, self._gripper_move_force)
            time.sleep(20)

        LOGGER.info("Switching back to NRT_CARTESIAN_MOTION_FORCE mode.")
        self._robot.SwitchMode(self._mode.NRT_CARTESIAN_MOTION_FORCE)
        self._robot.SetForceControlAxis([False, False, False, False, False, False])

    def get_ee_pose(self) -> np.ndarray:
        states = self._robot.states()
        pose_raw = states.flange_pose.copy() if self._action_frame == "flange" else states.tcp_pose.copy()
        pos = pose_raw[:3]
        qw, qx, qy, qz = pose_raw[3:]
        rot = st.Rotation.from_quat([qx, qy, qz, qw])
        return mat_to_certain_pose_type(pos_rot_to_mat(np.array(pos, dtype=np.float64), rot), "rotvec")

    def get_gripper_width(self) -> float:
        return float(self._gripper.states().width)

    def get_force(self) -> np.ndarray:
        try:
            states = self._gripper.states()
            if isinstance(states, dict):
                left = states.get("left_force", states.get("force", 0.0))
                right = states.get("right_force", states.get("force", left))
            else:
                left = getattr(states, "left_force", getattr(states, "force", 0.0))
                right = getattr(states, "right_force", getattr(states, "force", left))
            return np.asarray([left, right], dtype=np.float32)
        except Exception:
            return np.zeros((2,), dtype=np.float32)

    def _set_latest_target(self, target_tcp: list[float], target_time: float) -> None:
        with self._target_lock:
            self._latest_target_tcp = list(target_tcp)
            self._latest_target_time = float(target_time)

    def _action_pose_to_target_tcp(self, target_pose: np.ndarray) -> list[float]:
        target_pose_mat = pose_to_mat(target_pose)
        
        import os
        # 允许通过环境变量恢复到旧版本的直接发送（不进行 TCP offset 转换）
        if os.environ.get("FLEXIV_USE_DIRECT_EEF", "1") == "1":
            return _mat_to_tcp_pose7(target_pose_mat)

        if self._action_frame == "tcp":
            return _mat_to_tcp_pose7(target_pose_mat)

        states = self._robot.states()
        current_flange_mat = _pose7_to_mat(states.flange_pose)
        current_tcp_mat = _pose7_to_mat(states.tcp_pose)
        flange_to_tcp_mat = np.linalg.inv(current_flange_mat) @ current_tcp_mat
        target_tcp_mat = target_pose_mat @ flange_to_tcp_mat
        return _mat_to_tcp_pose7(target_tcp_mat)

    @staticmethod
    def _timestamp_to_seconds(timestamp: Any) -> float | None:
        values = _float_list(timestamp)
        if len(values) >= 2:
            return values[0] + values[1] * 1e-9
        if len(values) == 1:
            return values[0]
        return None

    def get_metric_snapshot(self) -> dict[str, Any]:
        states = self._robot.states()
        with self._target_lock:
            target_tcp = list(self._latest_target_tcp) if self._latest_target_tcp is not None else None
            target_time = self._latest_target_time

        return {
            "wall_time": time.time(),
            "robot_time": self._timestamp_to_seconds(getattr(states, "timestamp", None)),
            "tcp_pose": _float_list(getattr(states, "tcp_pose", None)),
            "tcp_vel": _float_list(getattr(states, "tcp_vel", None)),
            "flange_pose": _float_list(getattr(states, "flange_pose", None)),
            "flange_vel": _float_list(getattr(states, "flange_vel", None)),
            "q": _float_list(getattr(states, "q", None)),
            "theta": _float_list(getattr(states, "theta", None)),
            "dq": _float_list(getattr(states, "dq", None)),
            "dtheta": _float_list(getattr(states, "dtheta", None)),
            "tau": _float_list(getattr(states, "tau", None)),
            "tau_ext": _float_list(getattr(states, "tau_ext", None)),
            "ext_wrench_in_world": _float_list(getattr(states, "ext_wrench_in_world", None)),
            "ext_wrench_in_tcp": _float_list(getattr(states, "ext_wrench_in_tcp", None)),
            "ft_sensor_raw": _float_list(getattr(states, "ft_sensor_raw", None)),
            "dq_max": self._dq_max,
            "target_tcp": target_tcp,
            "target_time": target_time,
        }

    def exec_actions(
        self,
        actions: np.ndarray,
        timestamps: np.ndarray,
        target_force: float | None = None,
        *,
        should_reset: Callable[[], bool] | None = None,
    ) -> bool:
        receive_time = time.time()
        is_new = timestamps > receive_time
        new_actions = actions[is_new]
        new_timestamps = timestamps[is_new]

        for i in range(len(new_actions)):
            if should_reset is not None and should_reset():
                return False

            target_pose = new_actions[i, :6]
            target_width = float(new_actions[i, 6])

            target_tcp = self._action_pose_to_target_tcp(target_pose)
            if self._enable_safety_clip:
                from lerobot.common.robot_devices.robots.flexiv_safety import clip_target_pose_7d

                target_tcp = clip_target_pose_7d(target_tcp)
            self._set_latest_target(target_tcp, float(new_timestamps[i]))

            if self._dry_run:
                LOGGER.info("Dry run: skipping joint/gripper command for action %d/%d", i + 1, len(new_actions))
            else:
                self._robot.SendCartesianMotionForce(
                    target_tcp,
                    [0.0] * 6,
                    [0.0] * 6,
                    self._arm_max_linear_vel,
                    self._arm_max_angular_vel,
                    self._arm_max_linear_acc,
                    self._arm_max_angular_acc,
                )
                max_width = self._gripper.params().max_width
                safe_width = min(max(target_width, 0.001), max_width - 0.001)
                move_force = self._gripper_move_force if target_force is None else float(target_force)
                self._gripper.Move(safe_width, self._gripper_move_velocity, move_force)

            while True:
                if should_reset is not None and should_reset():
                    return False
                dt = new_timestamps[i] - time.time()
                if dt <= 0:
                    break
                time.sleep(min(dt, 0.05))

        return True


def _make_session_dir(args: Args) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return _repo_root() / args.record_root / args.task_name / timestamp


def _make_output_dir(session_dir: Path, episode_index: int) -> Path:
    return session_dir / f"episode_{episode_index:03d}"


def _create_camera_from_info(BaseCamera, camera_info: dict[str, Any]):
    camera_type = camera_info["type"]
    if camera_type == "mvs_cam":
        from perception.cameras.mvs_cam import MVSCamera

        return MVSCamera(
            {
                "serial": camera_info.get("serial"),
                "exposure": camera_info.get("exposure"),
            }
        )
    if camera_type == "realsense":
        return BaseCamera.create_camera(camera_type, camera_info.get("sn"))
    if camera_type in {"insta_front_fisheye", "insta360_front_fisheye"}:
        return BaseCamera.create_camera(
            camera_type,
            camera_info.get("serial"),
            fps=camera_info.get("fps"),
            resolution=camera_info.get("resolution"),
            capture_resolution=camera_info.get("capture_resolution"),
        )
    return BaseCamera.create_camera(
        camera_type,
        camera_info.get("camera_index"),
        camera_info.get("camera_v4l_path"),
        camera_info.get("fps"),
        camera_info.get("resolution"),
        camera_info.get("capture_resolution"),
    )


def _optional_record_camera_by_name(BaseCamera, config: dict[str, Any], name: str) -> Any | None:
    if name not in config:
        return None
    try:
        return _create_camera_from_info(BaseCamera, config[name])
    except Exception as exc:  # pragma: no cover - optional hardware path
        LOGGER.warning("\033[91mWARNING: Disabling optional recording camera %s after init failure: %s\033[0m", name, exc)
        return None

def _optional_record_camera_from_config(BaseCamera, config: dict[str, Any]) -> Any | None:
    for name in ("realsense", "record_realsense", "side", "head"):
        if name not in config or name == "wrist":
            continue
        try:
            return _create_camera_from_info(BaseCamera, config[name])
        except Exception as exc:  # pragma: no cover - optional hardware path
            LOGGER.warning("\033[91mWARNING: Disabling optional recording camera %s after init failure: %s\033[0m", name, exc)
            return None
    return None


def _init_cameras(camera_config_path: Path, use_tactile: bool) -> tuple[Any, Any | None, Any | None, Any | None, Any | None]:
    BaseCamera = _load_base_camera()
    config_path = camera_config_path

    if not use_tactile:
        LOGGER.info("use_tactile=False: using wrist camera only from %s", config_path)
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        if "wrist" not in config:
            raise KeyError(f"Camera config must contain a 'wrist' camera: {config_path}")
        cam_wrist = _create_camera_from_info(BaseCamera, config["wrist"])
        cam_realsense = _optional_record_camera_from_config(BaseCamera, config)
        cam_realsense_third = _optional_record_camera_by_name(BaseCamera, config, "realsense_third")
        return cam_wrist, cam_realsense, cam_realsense_third, None, None

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    if "wrist" not in config:
        raise KeyError(f"Camera config must contain a 'wrist' camera: {config_path}")
    missing_tactile = [name for name in ("left_tactile", "right_tactile") if name not in config]
    if missing_tactile:
        raise KeyError(
            "use_tactile=True requires tactile cameras in the camera config. "
            f"Missing {missing_tactile} from {config_path}"
        )
    cam_wrist = _create_camera_from_info(BaseCamera, config["wrist"])
    cam_realsense = _optional_record_camera_from_config(BaseCamera, config)
    cam_realsense_third = _optional_record_camera_by_name(BaseCamera, config, "realsense_third")
    cam_tactile_left = _create_camera_from_info(BaseCamera, config["left_tactile"])
    cam_tactile_right = _create_camera_from_info(BaseCamera, config["right_tactile"])
    return cam_wrist, cam_realsense, cam_realsense_third, cam_tactile_left, cam_tactile_right


def _init_recorder(
    output_dir: Path,
    episode_index: int,
    record_hz: float,
    cam_wrist,
    realsense_source,
    realsense_third_source,
    cam_tactile_left,
    cam_tactile_right,
    *,
    use_tactile: bool,
    save_states_json: bool,
) -> Recorder:
    _, init_wrist_img, _ = cam_wrist.read()

    if init_wrist_img is None:
        raise RuntimeError("Failed to fetch initial wrist frame for recorder setup")

    realsense_shape = WRIST_RECORD_SHAPE if realsense_source is not None else None
    realsense_third_shape = WRIST_RECORD_SHAPE if realsense_third_source is not None else None

    left_shape = None
    right_shape = None
    if use_tactile:
        if cam_tactile_left is None or cam_tactile_right is None:
            raise RuntimeError("use_tactile=True requires left and right tactile cameras")
        left_tactile_img, _ = _camera_get_bgr(cam_tactile_left)
        right_tactile_img, _ = _camera_get_bgr(cam_tactile_right)
        if left_tactile_img is None or right_tactile_img is None:
            raise RuntimeError("Failed to fetch initial tactile frames for recorder setup")
        left_shape = (left_tactile_img.shape[1], left_tactile_img.shape[0])
        right_shape = (right_tactile_img.shape[1], right_tactile_img.shape[0])

    return Recorder(
        output_dir=output_dir,
        episode_index=episode_index,
        video_fps=record_hz,
        wrist_shape=WRIST_RECORD_SHAPE,
        realsense_shape=realsense_shape,
        realsense_third_shape=realsense_third_shape,
        left_shape=left_shape,
        right_shape=right_shape,
        realsense_source=realsense_source,
        realsense_third_source=realsense_third_source,
        save_states_json=save_states_json,
    )


def _reset_to_init_pose(env: FlexivRealEnv, args: Args, reason: str) -> None:
    LOGGER.info("%s Moving robot to init pose: %s", reason, args.init_qpos)
    env.reset()
    LOGGER.info("Reset finished.")


def main(args: Args) -> None:
    from openpi_client import websocket_client_policy

    global _ACTIVE_RECORDER
    signal_module.signal(signal_module.SIGINT, _signal_handler)
    signal_module.signal(signal_module.SIGTERM, _signal_handler)

    control = RuntimeControl(args.control_file)
    recorder: Recorder | None = None
    observation_recorder: ObservationRecordingThread | None = None
    obs_thread: ObservationThread | None = None
    realsense_thread: LatestCameraThread | None = None
    metric_monitor: TrajectoryMetricMonitor | None = None
    tactile_calibration: TactileCalibration | None = None
    metric_started = False

    try:
        control.start()

        LOGGER.info("Connecting to remote policy server at ws://%s:%d", args.host, args.port)
        if args.control_file is not None:
            LOGGER.info("Interactive control file: %s", args.control_file)
        prompt = args.prompt or args.task_name
        robot_dt = 1.0 / args.ctrl_freq
        LOGGER.info(
            "Control config: ctrl_freq=%.3fHz, record_hz=%.3fHz, robot_dt=%.4fs, steps_per_inference=%d, action_latency=%.4fs, "
            "force_predict=%s, force_guide=%s, infer_force_control=%s",
            args.ctrl_freq,
            args.record_hz,
            robot_dt,
            args.steps_per_inference,
            args.action_latency,
            args.force_predict,
            args.force_guide,
            args.infer_force_control,
        )
        metric_config = _make_trajectory_cost_config(args)
        LOGGER.info(
            "Trajectory cost config: monitor_hz=%.1f, T_ref=%.3f, v_ref=%.3f, weights=(%.2f, %.2f, %.2f)",
            metric_config.monitor_hz,
            metric_config.t_ref,
            metric_config.v_ref,
            metric_config.w_e,
            metric_config.w_m,
            metric_config.w_c,
        )

        camera_config_path = args.camera_config_path
        if not camera_config_path.is_absolute():
            camera_config_path = (_repo_root() / camera_config_path).resolve()
        if not camera_config_path.exists():
            camera_config_path = _default_camera_config_path()

        policy = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
        LOGGER.info("Server metadata: %s", policy.get_server_metadata())

        LOGGER.info("Initializing cameras from %s (use_tactile=%s)", camera_config_path, args.use_tactile)
        cam_wrist, cam_realsense, cam_realsense_third, cam_tactile_left, cam_tactile_right = _init_cameras(camera_config_path, args.use_tactile)
        realsense_thread = LatestCameraThread(cam_realsense, "realsense") if cam_realsense is not None else None
        realsense_third_thread = LatestCameraThread(cam_realsense_third, "realsense_third") if cam_realsense_third is not None else None
        if args.use_tactile:
            if args.tactile_calibration:
                LOGGER.info(
                    "Computing live tactile calibration from camera frames (attempts=%d per side).",
                    args.tactile_calibration_attempts,
                )
                tactile_calibration = _compute_live_tactile_calibration(
                    cam_tactile_left,
                    cam_tactile_right,
                    args.tactile_calibration_attempts,
                )
            else:
                LOGGER.warning("Tactile calibration disabled; raw tactile frames will be used.")

        LOGGER.info("Initializing Flexiv environment at %s", args.robot_ip)
        env = FlexivRealEnv(args.init_qpos, robot_ip=args.robot_ip, local_ip=args.local_ip, dry_run=args.dry_run)
        _reset_to_init_pose(env, args, "Startup reset.")

        if realsense_thread is not None:
            realsense_thread.start()
        if realsense_third_thread is not None:
            realsense_third_thread.start()

        session_dir = _make_session_dir(args)
        if tactile_calibration is not None:
            _save_tactile_calibration(session_dir, tactile_calibration)
        episode_index = 0
        LOGGER.info("Session recording directory: %s", session_dir)
        LOGGER.info("Ready. Press SPACE to start/stop one episode, r to reset, Ctrl+C to exit.")
        while True:
            while not control.consume_start():
                if control.consume_reset():
                    _reset_to_init_pose(env, args, "Reset requested while idle.")
                control.consume_stop()
                time.sleep(0.05)

            if control.consume_reset():
                _reset_to_init_pose(env, args, "Reset requested before episode start.")

            control.consume_stop()
            output_dir = _make_output_dir(session_dir, episode_index)
            LOGGER.info(
                "\033[92mStart command accepted. Initializing episode %03d recorder in %s\033[0m",
                episode_index,
                output_dir,
            )
            control.emit_event("episode_started", output_dir)
            recorder = _init_recorder(
                output_dir,
                episode_index,
                args.record_hz,
                cam_wrist,
                realsense_thread,
                realsense_third_thread,
                cam_tactile_left,
                cam_tactile_right,
                use_tactile=args.use_tactile,
                save_states_json=args.save_states_json,
            )
            _ACTIVE_RECORDER = recorder

            obs_thread = ObservationThread(
                cam_wrist,
                cam_tactile_left,
                cam_tactile_right,
                env,
                maxlen=args.obs_horizon,
                tactile_calibration=tactile_calibration,
            )
            obs_thread.start()
            observation_recorder = ObservationRecordingThread(obs_thread, recorder, args.record_hz)
            observation_recorder.start()
            LOGGER.info("Continuous observation recording started at %.3fHz.", args.record_hz)

            metric_monitor = TrajectoryMetricMonitor(env, metric_config)
            recorder.attach_metric_monitor(metric_monitor, metric_config)
            metric_started = False
            step = 0

            try:
                LOGGER.info("Waiting for observation queue to fill (obs_horizon=%d)", args.obs_horizon)
                episode_stopped_before_inference = False
                while obs_thread.get_obs(args.obs_horizon) is None:
                    if control.stop_requested.is_set():
                        LOGGER.info("Stop requested before observation queue filled.")
                        episode_stopped_before_inference = True
                        break
                    if control.consume_reset():
                        _reset_to_init_pose(env, args, "Reset requested while filling observations.")
                    time.sleep(0.01)
                if not episode_stopped_before_inference:
                    LOGGER.info("Observation queue filled. Starting inference loop.")

                while not control.stop_requested.is_set():
                    if episode_stopped_before_inference:
                        break
                    if control.consume_reset():
                        _reset_to_init_pose(env, args, "Reset requested.")
                        continue

                    loop_start = time.time()
                    frames = obs_thread.get_obs(args.obs_horizon)
                    if frames is None:
                        time.sleep(0.01)
                        continue

                    latest_frame = frames[-1]
                    recorder.record_frame(latest_frame, model_input=True)
                    observation = _make_policy_observation(latest_frame, prompt, args.use_tactile)
                    client_infer_start = time.monotonic()
                    action_result = policy.infer(observation)
                    client_policy_roundtrip_ms = (time.monotonic() - client_infer_start) * 1000.0
                    action_chunk = np.asarray(action_result["actions"], dtype=np.float64)
                    decoded_actions = decode_handcap_action_chunk(action_chunk)
                    policy_chunk_size = len(decoded_actions)

                    if control.consume_stop():
                        LOGGER.info("Stop requested before action execution.")
                        break
                    if control.consume_reset():
                        _reset_to_init_pose(env, args, "Reset requested before action execution.")
                        continue

                    # Convert relative SE(3) increments back to absolute physical target poses.
                    physical_actions = []
                    current_eepose = latest_frame["eepose"]
                    for action in decoded_actions:
                        abs_phys_pose = get_real_umi_inference_action(action[:6], current_eepose)
                        physical_actions.append(np.concatenate([abs_phys_pose, action[6:7]], axis=-1))
                    physical_actions = np.array(physical_actions)
                    if args.steps_per_inference > 0:
                        physical_actions = physical_actions[: args.steps_per_inference]

                    action_timestamps = (
                        (1 + np.arange(len(physical_actions), dtype=np.float64)) * robot_dt
                        + time.time()
                    )

                    force_pred = action_chunk[0, 10:12] if action_chunk.shape[-1] >= 12 else np.zeros((2,), dtype=np.float64)
                    if args.infer_force_control and (args.force_predict or args.force_guide):
                        target_force = max(1.0, float(np.mean(force_pred)))
                    else:
                        target_force = None

                    if not metric_started:
                        metric_monitor.start()
                        metric_started = True

                    actions_completed = env.exec_actions(
                        physical_actions,
                        action_timestamps,
                        target_force=target_force,
                        should_reset=lambda: control.reset_requested.is_set() or control.stop_requested.is_set(),
                    )
                    if not actions_completed:
                        if control.consume_stop():
                            LOGGER.info("Stop requested during action execution.")
                            break
                        if control.consume_reset():
                            _reset_to_init_pose(env, args, "Reset requested during action execution.")
                            continue
                    if control.consume_stop():
                        LOGGER.info("Stop requested after action execution.")
                        break
                    if control.consume_reset():
                        _reset_to_init_pose(env, args, "Reset requested during action execution.")
                        continue

                    inference_latency = time.time() - loop_start
                    step += 1
                    recorder.append_state(
                        step=step,
                        inference_latency=inference_latency,
                        latest_frame=latest_frame,
                        action_chunk=action_chunk,
                        client_timing={"policy_roundtrip_ms": client_policy_roundtrip_ms},
                        server_timing=action_result.get("server_timing"),
                        policy_timing=action_result.get("policy_timing"),
                    )

                    first_target = physical_actions[0]
                    force_obs = _prepare_force(latest_frame.get("force"))
                    exec_horizon = len(physical_actions) * robot_dt
                    obs_timing = latest_frame.get("timing") if isinstance(latest_frame.get("timing"), dict) else {}
                    camera_read_ms = float(obs_timing.get("camera_read_ms", 0.0))
                    obs_total_ms = float(obs_timing.get("obs_total_ms", 0.0))
                    robot_state_ms = float(obs_timing.get("robot_state_ms", 0.0))
                    preprocess_ms = float(obs_timing.get("preprocess_ms", 0.0))
                    server_timing = action_result.get("server_timing") or {}
                    policy_timing = action_result.get("policy_timing") or {}
                    server_infer_ms = float(server_timing.get("infer_ms", 0.0))
                    model_forward_ms = float(policy_timing.get("infer_ms", 0.0))
                    if step % 5 == 0:
                        LOGGER.info(
                            "\033[92m[step=%d] infer=%.1fms policy_chunk=%d exec_steps=%d eef_xyz=(%.3f, %.3f, %.3f) "
                            "exec_horizon=%.2fs gripper=%.4f force=(%.2f, %.2f) target_xyz=(%.3f, %.3f, %.3f) "
                            "target_gripper=%.4f pred_force=(%.2f, %.2f) "
                            "timing_ms(camera_read=%.1f obs_total=%.1f robot_state=%.1f preprocess=%.1f "
                            "model_forward=%.1f server_infer=%.1f client_roundtrip=%.1f)\033[0m",
                            step,
                        inference_latency * 1000.0,
                        policy_chunk_size,
                        len(physical_actions),
                        float(current_eepose[0]),
                        float(current_eepose[1]),
                        float(current_eepose[2]),
                        exec_horizon,
                        float(latest_frame["gripper_width"]),
                        float(force_obs[0]),
                        float(force_obs[1]),
                        float(first_target[0]),
                        float(first_target[1]),
                        float(first_target[2]),
                        float(first_target[6]),
                        float(force_pred[0]),
                        float(force_pred[1]),
                        camera_read_ms,
                        obs_total_ms,
                        robot_state_ms,
                        preprocess_ms,
                        model_forward_ms,
                        server_infer_ms,
                        client_policy_roundtrip_ms,
                    )

                    if step % 2 == 0:
                        metric_samples_snapshot = metric_monitor.snapshot()
                        metric_summary = compute_trajectory_cost(metric_samples_snapshot, metric_config)
                        if metric_summary.get("valid"):
                            LOGGER.info(
                                "\033[92m[step=%d] Current Trajectory Cost: J=%.3f (J_e=%.3f, J_m=%.3f, J_c=%.3f)\033[0m",
                                step,
                                metric_summary.get("J", 0.0),
                                metric_summary.get("J_e", 0.0),
                                metric_summary.get("J_m", 0.0),
                                metric_summary.get("J_c", 0.0),
                            )
            finally:
                control.consume_stop()
                if metric_monitor is not None:
                    metric_monitor.stop()
                    if metric_started:
                        metric_monitor.join(timeout=2.0)
                    if metric_monitor.error is not None:
                        LOGGER.warning("Trajectory metric monitor stopped after sampling error: %s", metric_monitor.error)
                    metric_monitor = None
                    metric_started = False
                if observation_recorder is not None:
                    observation_recorder.stop()
                    observation_recorder.join(timeout=2.0)
                    observation_recorder = None
                if obs_thread is not None:
                    obs_thread.stop()
                    obs_thread = None
                if recorder is not None:
                    recorder.flush()
                    recorder = None
                _ACTIVE_RECORDER = None
                control.emit_event("episode_finished", output_dir)
                LOGGER.info("Episode finished: %s", output_dir)
                decision = control.wait_for_save_decision()
                if decision == "keep":
                    episode_index += 1
                    LOGGER.info("Episode kept. Next episode index will be %03d.", episode_index)
                else:
                    LOGGER.info("Episode discarded. Next episode will reuse index %03d.", episode_index)
                LOGGER.info("Ready. Press SPACE to start episode %03d, r to reset, Ctrl+C to exit.", episode_index)
    finally:
        control.stop()
        if metric_monitor is not None:
            metric_monitor.stop()
            if metric_started:
                metric_monitor.join(timeout=2.0)
            if metric_monitor.error is not None:
                LOGGER.warning("Trajectory metric monitor stopped after sampling error: %s", metric_monitor.error)
        if observation_recorder is not None:
            observation_recorder.stop()
            observation_recorder.join(timeout=2.0)
        if obs_thread is not None:
            obs_thread.stop()
        if realsense_thread is not None:
            realsense_thread.stop()
        if realsense_third_thread is not None:
            realsense_third_thread.stop()
        if recorder is not None:
            recorder.flush()
        if _ACTIVE_RECORDER is recorder:
            _ACTIVE_RECORDER = None


if __name__ == "__main__":
    import tyro

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        force=True,
    )
    main(tyro.cli(Args))
