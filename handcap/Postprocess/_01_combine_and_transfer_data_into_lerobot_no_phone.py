"""Combine HTC-only HandCap data into a LeRobot v3 dataset.

This no-phone variant is for collections made by htc.py, where HTC pose is
continuous and phone pose/camera/depth are intentionally absent.
"""

import os
import cv2
import sys
import shutil
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lerobot30_env"))

import json
import numpy as np 
from pathlib import Path
from umi_utils.pose_util import *
from argparse import ArgumentParser
from scipy.interpolate import interp1d
from huggingface_hub.constants import HF_HOME
from tactile_calibration.calibrate_tactile import compute_calibration_params, apply_calibration
from lerobot.datasets.video_utils import VideoEncodingManager, encode_video_frames
from lerobot.datasets.image_writer import safe_stop_image_writer
from lerobot.datasets.dataset_metadata import CODEBASE_VERSION as LEROBOT_CODEBASE_VERSION
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata, LeRobotDataset

if LEROBOT_CODEBASE_VERSION != "v3.0":
    raise RuntimeError(f"FATAL ERROR: The imported LeRobot engine is currently at version {LEROBOT_CODEBASE_VERSION}, but v3.0 is STRICTLY REQUIRED! Please check your local lerobot30 copies.")

from scipy.spatial.transform import Rotation


SCRIPT_DIR = Path(__file__).resolve().parent
HANDCAP_ROOT = SCRIPT_DIR.parent


def resolve_existing_path(path_like, *, base_dirs):
    path = Path(path_like).expanduser()
    if path.is_absolute():
        if path.exists():
            return path
        raise FileNotFoundError(f"Path does not exist: {path}")

    candidates = [Path.cwd() / path]
    candidates.extend(Path(base_dir) / path for base_dir in base_dirs)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    searched = "\n".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not resolve relative path {path_like!r}. Searched:\n{searched}")


def extract_capture_id(*names):
    for name in names:
        base = os.path.basename(str(name).rstrip(os.sep))
        for part in reversed(base.split("_")):
            digits = "".join(ch for ch in part if ch.isdigit())
            if digits:
                return digits
    return None


def load_angle_calibration_map(map_path):
    resolved_map_path = resolve_existing_path(map_path, base_dirs=[SCRIPT_DIR, HANDCAP_ROOT])
    with open(resolved_map_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    if not isinstance(mapping, dict):
        raise ValueError(f"Angle calibration map must be a JSON object: {resolved_map_path}")
    return {str(k): str(v) for k, v in mapping.items()}, resolved_map_path


def resolve_angle_calibration_path(capture_id, angle_calib_map, map_path):
    if capture_id not in angle_calib_map:
        available = ", ".join(sorted(angle_calib_map))
        raise KeyError(
            f"No angle calibration entry for capture id {capture_id!r} in {map_path}. "
            f"Available ids: {available}"
        )
    return resolve_existing_path(angle_calib_map[capture_id], base_dirs=[HANDCAP_ROOT, SCRIPT_DIR, map_path.parent])


########################################### interpolator ###########################################

def resize_image_high_quality(image, target_w, target_h, is_depth=False):
    """Resize with interpolation chosen to avoid unnecessary blur."""
    h, w = image.shape[:2]
    if (h, w) == (target_h, target_w):
        return image

    if is_depth:
        interpolation = cv2.INTER_NEAREST
    elif target_w < w or target_h < h:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_LANCZOS4

    return cv2.resize(image, (target_w, target_h), interpolation=interpolation)


def resize_tactile_to_model_input(image, target_size=(224, 224)):
    """Keep tactile data geometry aligned with collection and model input."""
    target_w, target_h = target_size
    return resize_image_high_quality(image, target_w, target_h)


def resize_wrist_to_model_input(image, target_size=(224, 224)):
    """Directly resize raw wrist frames to model input size without padding."""
    target_w, target_h = target_size
    if image.shape[:2] == (target_h, target_w):
        return image
    return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)


def probe_video_encoding(vcodec, pix_fmt, video_gop, video_crf):
    """Try a tiny encode to catch missing H.264/x264 packages before processing data."""
    import tempfile

    probe_dir = Path(tempfile.mkdtemp(prefix="handcap_video_probe_"))
    try:
        for frame_idx, value in enumerate((32, 160)):
            frame = np.full((32, 48, 3), value, dtype=np.uint8)
            cv2.imwrite(str(probe_dir / f"frame-{frame_idx:06d}.png"), frame)
        encode_video_frames(
            probe_dir,
            probe_dir / "probe.mp4",
            fps=24,
            vcodec=vcodec,
            pix_fmt=pix_fmt,
            g=video_gop,
            crf=video_crf,
            overwrite=True,
        )
        return True, None
    except Exception as e:
        return False, e
    finally:
        shutil.rmtree(probe_dir, ignore_errors=True)


def ensure_working_video_encoding(args):
    """Select the requested encoder, or fall back to the closest compatible H.264 path."""
    candidates = [(args.vcodec, args.video_pix_fmt)]
    if args.video_pix_fmt != "yuv420p":
        candidates.append((args.vcodec, "yuv420p"))
    if args.vcodec == "h264":
        candidates.extend([
            ("libx264", args.video_pix_fmt),
            ("libx264", "yuv420p"),
            ("h264_videotoolbox", "yuv420p"),
        ])
    elif args.vcodec == "libx264":
        candidates.extend([
            ("h264", args.video_pix_fmt),
            ("h264", "yuv420p"),
            ("h264_videotoolbox", "yuv420p"),
        ])
    elif args.vcodec == "auto":
        candidates.extend([
            ("h264", "yuv420p"),
            ("libx264", "yuv420p"),
        ])
    candidates.extend([
        ("h264_videotoolbox", "yuv420p"),
        ("libsvtav1", "yuv420p"),
    ])

    seen = set()
    errors = []
    for vcodec, pix_fmt in candidates:
        key = (vcodec, pix_fmt)
        if key in seen:
            continue
        seen.add(key)

        ok, err = probe_video_encoding(vcodec, pix_fmt, args.video_gop, args.video_crf)
        if ok:
            if (vcodec, pix_fmt) != (args.vcodec, args.video_pix_fmt):
                print(
                    "Video encoder fallback selected: "
                    f"{args.vcodec}/{args.video_pix_fmt} -> {vcodec}/{pix_fmt}. "
                    "This usually means the requested H.264 package or pixel format is unavailable."
                )
            else:
                print(f"Video encoder check passed: {vcodec}/{pix_fmt}, crf={args.video_crf}, gop={args.video_gop}")
            args.vcodec = vcodec
            args.video_pix_fmt = pix_fmt
            return

        errors.append(f"{vcodec}/{pix_fmt}: {err}")

    raise RuntimeError(
        "No working video encoder found. Install a PyAV/FFmpeg build with x264, for example "
        "`conda install -n handcap -c conda-forge av ffmpeg x264`, then rerun. "
        "Probe failures:\n" + "\n".join(errors[-8:])
    )


class WrappedInterp1d:
    def __init__(self, X, Y, extend_boundary=None):
        sort_id = np.argsort(X)
        X = np.array(X)[sort_id]
        Y = [Y[i] for i in sort_id]
        
        self.traj_interp = interp1d(X, Y, axis=0, bounds_error=True)

        extend_boundary = 0 if extend_boundary is None else extend_boundary
        self.left_min, self.left_max = X[0]-extend_boundary, X[0]
        self.right_min, self.right_max = X[-1], X[-1]+extend_boundary
        self.left_val = Y[0]
        self.right_val = Y[-1]
        
    def __call__(self, x):
        if self.left_min < x < self.left_max:
            return self.left_val
        elif self.right_min < x < self.right_max:
            return self.right_val
        else:
            return self.traj_interp(x)


def PostProcess_quat(pose):
    xyz = pose[:3]
    quat = pose[3:]
    new_quat = [quat[3], quat[0], quat[1], quat[2]]
    return np.array([xyz[0], xyz[1], xyz[2], new_quat[0], new_quat[1], new_quat[2], new_quat[3]])


def compute_pose_transform(raw_pose):
    raw_pose = np.array(raw_pose)
    pose_mat = pose_to_mat(quat_to_rot(raw_pose))
    
    additional_transform_matrix = np.eye(4)
    additional_R_y = Rotation.from_euler('y', 90, degrees=True).as_matrix()
    additional_R_z = Rotation.from_euler('z', 180, degrees=True).as_matrix()
    transform_mat = additional_R_y @ additional_R_z
    additional_transform_matrix[:3, :3] = transform_mat
    pose_mat_out = pose_mat @ additional_transform_matrix
    
    offset_mm = np.array([107.68, 39.0, -76.86])
    offset_m = offset_mm / 1000.0

    # Offset is measured in the vertical-down home EEF frame:
    # X/back, Y/left, Z/down. Convert it to the world frame
    # X/front, Y/left, Z/up before applying it to the position.
    home_eef_axes_in_world = np.column_stack([
        np.array([-1.0, 0.0, 0.0]),  # EEF X/back
        np.array([0.0, 1.0, 0.0]),   # EEF Y/left
        np.array([0.0, 0.0, -1.0]),  # EEF Z/down
    ])
    pose_mat_out[:3, 3] += home_eef_axes_in_world @ offset_m
    
    return mat_to_certain_pose_type(pose_mat_out, pose_type="rotvec")

features =  {
        "start_pose": {
            "dtype": "float32",
            "shape": (6,), 
            "names": ["x", "y", "z", "qx", "qy", "qz"]  
        },
        "end_pose": {
            "dtype": "float32",
            "shape": (6,), 
            "names": ["x", "y", "z", "qx", "qy", "qz"] 
        },
        "action": {
            "dtype": "float32",
            "shape": (10,), 
            "names": ["x", "y", "z" "qx", "qy", "qz", "gripper", "none1", "none2", "none3"] 
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (10,),
            "names": ["x", "y", "z", "qx", "qy", "qz", "gripper", "none1", "none2","none3"]
        },
        "observation.images.wrist": {
            "dtype": "video",
            "shape": (224, 224, 3),
            "names": ["height", "width", "channels"],
            "info": None
        },
        "observation.tactiles.left": {
            "dtype": "video",
            "shape": (224, 224, 3),
            "names": ["height", "width", "channels"],
            "info": None
        },
        "observation.tactiles.right": {
            "dtype": "video",
            "shape": (224, 224, 3),
            "names": ["height", "width", "channels"],
            "info": None
        },
        "observation.forces.left": {
            "dtype": "float32",
            "shape": (1,),
            "names": None
        },
        "observation.forces.right": {
            "dtype": "float32",
            "shape": (1,),
            "names": None
        },
        "timestamp": {
            "dtype": "float32",
            "shape": (1,),
            "names": None
        },
        "frame_index": {
            "dtype": "int64",
            "shape": (1,),
            "names": None
        },
        "episode_index": {
            "dtype": "int64",
            "shape": (1,),
            "names": None
        },
        "index": {
            "dtype": "int64",
            "shape": (1,),
            "names": None
        },
        "task_index": {
            "dtype": "int64",
            "shape": (1,),
            "names": None
        }
}

def find_closest_timestamp(target_time, timestamps, values, current_idx, threshold=0.1):
    if len(timestamps) == 0:
        return None, current_idx

    start_idx = max(0, current_idx - 10000)
    best_idx = -1
    
    for i in range(start_idx, len(timestamps)):
        ts = timestamps[i]
        if i > 0 and ts > target_time:
            prev_ts = timestamps[i-1]
            if abs(target_time - prev_ts) <= abs(target_time - ts):
                best_idx = i - 1
            else:
                best_idx = i
            break
    else:
        best_idx = len(timestamps) - 1

    if best_idx == -1:
        raise ValueError(f"Could not find a suitable timestamp for target {target_time}")

    if abs(target_time - timestamps[best_idx]) > threshold:
        if current_idx > 100000:
            start_idx = current_idx - 100000
        else:
            start_idx = 0
            
        best_idx = -1
        
        for i in range(start_idx, len(timestamps)):
            ts = timestamps[i]
            if i > 0 and ts > target_time:
                prev_ts = timestamps[i-1]
                if abs(target_time - prev_ts) <= abs(target_time - ts):
                    best_idx = i - 1
                else:
                    best_idx = i
                break
        else:
            best_idx = len(timestamps) - 1

        if best_idx == -1:
            raise ValueError(f"Could not find a suitable timestamp for target {target_time}")
        if abs(target_time - timestamps[best_idx]) > threshold:
            return None, best_idx

    return values[best_idx], best_idx


@safe_stop_image_writer
def record_loop(dataset, args):


    file_list = os.listdir(args.data_root)
    angle_calib_map, angle_calib_map_path = load_angle_calibration_map(args.angle_calib_map)
    print(
        f"[AngleCalib] Loaded id->angle calibration map from {angle_calib_map_path} "
        f"({len(angle_calib_map)} ids: {', '.join(sorted(angle_calib_map))})"
    )
    tracker_file_list = []
    if args.cap_id.lower() == "all":
        tracker_file_list = [f for f in file_list if f.startswith("htc")]
    else:
        cap_ids = args.cap_id.split(" ")
        for tf in cap_ids:
            tracker_file_list += [f for f in file_list if f.startswith("htc") and tf in f]
        
    tracker_file_list = sorted(list(set(tracker_file_list)))
    
    if not tracker_file_list:
        requested_cap_ids = args.cap_id if args.cap_id.lower() == "all" else cap_ids
        print(f"Error: No matching data folders found in {args.data_root} for cap_ids {requested_cap_ids}.")
        return

    handcap_file_list = ["handcap_" + f.split("_")[-1] for f in tracker_file_list]
    
    def preload_data(dataroot, file_list, prefix, value_key, time_key, is_npz=False):
        all_values = []
        all_times = []
        for file_name in sorted(file_list, key=lambda f: int(f.split('_')[-1].split('.')[0]) if '_' in f else 0):
            file_path = os.path.join(dataroot, file_name)
            if is_npz:
                data = np.load(file_path)
                all_values.extend(data[value_key])
                all_times.extend(data[time_key])
            else:
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse {file_path}: {e}")
                    continue
                all_values.extend(data[value_key])
                all_times.extend(data[time_key])
        return np.array(all_times), all_values
    
    def preload_angle_data(dataroot, file_list, prefix, value_key, time_key):
        return preload_data(dataroot, file_list, prefix, value_key, time_key)

    def preload_video_data(dataroot, file_list, prefix, mp4_ext=".mp4"):
        all_times = []
        all_video_paths = []
        all_frame_indices = []
        for file_name in sorted(file_list, key=lambda f: int(f.split('_')[-1].split('.')[0]) if '_' in f else 0):
            json_path = os.path.join(dataroot, file_name)
            video_path = os.path.join(dataroot, file_name.replace('.json', mp4_ext))
            try:
                with open(json_path, 'r') as f:
                    timestamps = json.load(f)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse {json_path}: {e}")
                continue
            if isinstance(timestamps, dict) and 'timestamps' in timestamps:
                timestamps = timestamps['timestamps']

            if not os.path.exists(video_path):
                print(f"Warning: Missing data file for {json_path}: {video_path}. Skipping this chunk.")
                continue

            valid_len = len(timestamps)
            if mp4_ext == ".mp4":
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    print(f"Warning: Could not open video file {video_path}. Skipping this chunk.")
                    continue
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                valid_len = min(valid_len, frame_count)
                if valid_len < len(timestamps):
                    print(
                        f"Warning: Truncating {json_path}: "
                        f"{len(timestamps)} timestamps but {frame_count} video frames."
                    )
            elif mp4_ext == ".npz":
                try:
                    with np.load(video_path) as data:
                        key = "depths" if "depths" in data else list(data.keys())[0]
                        frame_count = len(data[key])
                except Exception as e:
                    print(f"Warning: Could not inspect npz file {video_path}: {e}. Skipping this chunk.")
                    continue
                valid_len = min(valid_len, frame_count)
                if valid_len < len(timestamps):
                    print(
                        f"Warning: Truncating {json_path}: "
                        f"{len(timestamps)} timestamps but {frame_count} npz frames."
                    )

            if valid_len <= 0:
                print(f"Warning: No readable frames for {json_path}. Skipping this chunk.")
                continue

            all_times.extend(timestamps[:valid_len])
            all_video_paths.extend([video_path] * valid_len)
            all_frame_indices.extend(range(valid_len))
        combined_values = list(zip(all_video_paths, all_frame_indices))
        return np.array(all_times), combined_values
    
    video_caps = {}
    global_calib_params = None
    global_calib_source = None

    def extract_episode_num(name):
        nums = ''.join(filter(str.isdigit, name))
        return int(nums) if nums else 999999

    def get_frame(video_path, frame_idx):
        if video_path not in video_caps:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Error opening video file: {video_path}")
            video_caps[video_path] = cap
        
        cap = video_caps[video_path]
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            cap.release()
            del video_caps[video_path]
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Error reopening video file: {video_path}")
            video_caps[video_path] = cap
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            raise ValueError(f"Failed to read frame {frame_idx} from {video_path} on retry.")
                
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def compute_first_valid_tactile_calibration(tactile_values, side_name, rng):
        candidate_indices = np.arange(len(tactile_values))
        rng.shuffle(candidate_indices)
        last_error = None
        for candidate_idx in candidate_indices:
            video_path, frame_idx = tactile_values[int(candidate_idx)]
            try:
                frame_bgr = cv2.cvtColor(get_frame(video_path, frame_idx), cv2.COLOR_RGB2BGR)
                calib_params = compute_calibration_params(frame_bgr)
                source = {
                    "video_path": video_path,
                    "frame_idx": int(frame_idx),
                    "candidate_idx": int(candidate_idx),
                }
                print(
                    f"{side_name} tactile calibration succeeded using "
                    f"candidate {candidate_idx}: {video_path} frame {frame_idx}"
                )
                return calib_params, source
            except ValueError as e:
                last_error = e
                continue

        raise ValueError(
            f"Could not compute {side_name} tactile calibration from "
            f"{len(tactile_values)} candidate frames. Last error: {last_error}"
        )

    def find_run_tactile_calibration():
        rng = np.random.default_rng(args.calibration_seed)
        candidate_episodes = []
        for handcap_data_root in handcap_file_list:
            handcap_full_path = os.path.join(args.data_root, handcap_data_root)
            try:
                episode_folders = [
                    d for d in os.listdir(handcap_full_path)
                    if os.path.isdir(os.path.join(handcap_full_path, d))
                ]
            except Exception as e:
                print(f"Warning: Could not inspect {handcap_full_path} for calibration: {e}")
                continue

            episode_folders.sort(key=lambda x: extract_episode_num(x))
            for ep_folder in episode_folders:
                if extract_episode_num(ep_folder) < args.start_episode_idx:
                    continue
                candidate_episodes.append((handcap_data_root, os.path.join(handcap_full_path, ep_folder)))

        if not candidate_episodes:
            raise ValueError("No eligible episodes found for tactile calibration.")

        episode_indices = np.arange(len(candidate_episodes))
        rng.shuffle(episode_indices)
        failures = []

        for episode_idx in episode_indices:
            handcap_data_root, ep_root = candidate_episodes[int(episode_idx)]
            tactile_dataroot = os.path.join(ep_root, "tactile")
            wrist_dataroot = os.path.join(ep_root, "wrist")
            if not os.path.exists(wrist_dataroot):
                tactile_dataroot = ep_root

            try:
                left_tactile_file_list = [
                    f for f in os.listdir(tactile_dataroot)
                    if f.startswith("lefttactile_") and f.endswith(".json")
                ]
                right_tactile_file_list = [
                    f for f in os.listdir(tactile_dataroot)
                    if f.startswith("righttactile_") and f.endswith(".json")
                ]
            except Exception as e:
                failures.append(f"{ep_root}: missing tactile structure ({e})")
                continue

            _, left_tactile_values = preload_video_data(
                tactile_dataroot, left_tactile_file_list, "lefttactile_"
            )
            _, right_tactile_values = preload_video_data(
                tactile_dataroot, right_tactile_file_list, "righttactile_"
            )

            if len(left_tactile_values) == 0 or len(right_tactile_values) == 0:
                failures.append(
                    f"{ep_root}: no usable tactile frames "
                    f"(left={len(left_tactile_values)}, right={len(right_tactile_values)})"
                )
                continue

            try:
                left_calib_params, left_calib_source = compute_first_valid_tactile_calibration(
                    left_tactile_values, "left", rng
                )
                right_calib_params, right_calib_source = compute_first_valid_tactile_calibration(
                    right_tactile_values, "right", rng
                )
                source = {
                    "handcap_data_root": handcap_data_root,
                    "episode": ep_root,
                    "left_source": left_calib_source,
                    "right_source": right_calib_source,
                }
                return (left_calib_params, right_calib_params), source
            except ValueError as e:
                failures.append(f"{ep_root}: {e}")

        raise ValueError(
            "Failed to find any episode/frame pair that can compute tactile calibration. "
            "Recent failures:\n" + "\n".join(failures[-10:])
        )

    if getattr(args, 'use_refer_tactile', False):
        print("Reference tactile mode: skipping tactile calibration search.")
    else:
        global_calib_params, global_calib_source = find_run_tactile_calibration()
        calib_saved_path = os.path.join(args.data_root, "tactile_calib.json")
        calib_data = {
            "left_H": global_calib_params[0][0].tolist(),
            "left_mask": global_calib_params[0][1].tolist(),
            "left_w": int(global_calib_params[0][2]),
            "left_h": int(global_calib_params[0][3]),
            "right_H": global_calib_params[1][0].tolist(),
            "right_mask": global_calib_params[1][1].tolist(),
            "right_w": int(global_calib_params[1][2]),
            "right_h": int(global_calib_params[1][3]),
            "source": global_calib_source,
        }
        with open(calib_saved_path, 'w') as f:
            json.dump(calib_data, f)
        print(f"Calibration parameters selected for this run and saved to {calib_saved_path}")

        for cap in video_caps.values():
            cap.release()
        video_caps.clear()
    
    for batch_idx, (handcap_data_root, tracker_root) in enumerate(zip(handcap_file_list, tracker_file_list)):
        
        capture_id = extract_capture_id(handcap_data_root, tracker_root)
        if capture_id is None:
            raise ValueError(f"Could not extract capture id from {handcap_data_root!r} / {tracker_root!r}")
        angle_calib_path = resolve_angle_calibration_path(capture_id, angle_calib_map, angle_calib_map_path)
        angle_calib_entry = angle_calib_map[capture_id]
        print(
            "[AngleCalib] Using calibration:\n"
            f"  batch_idx={batch_idx}\n"
            f"  capture_id={capture_id}\n"
            f"  handcap_root={handcap_data_root}\n"
            f"  tracker_root={tracker_root}\n"
            f"  map_entry={angle_calib_entry}\n"
            f"  resolved_path={angle_calib_path}"
        )
        with open(angle_calib_path, "r") as fp:
            width_calib = json.load(fp)["calibration"]
            widths, values = zip(*width_calib)
            width_arr = np.asarray(widths, dtype=np.float64)
            value_arr = np.asarray(values, dtype=np.float64)
            print(
                "[AngleCalib] Loaded calibration table:\n"
                f"  capture_id={capture_id}\n"
                f"  points={len(width_calib)}\n"
                f"  width_range=({float(np.min(width_arr)):.6f}, {float(np.max(width_arr)):.6f})\n"
                f"  raw_value_range=({float(np.min(value_arr)):.6f}, {float(np.max(value_arr)):.6f})"
            )
            
            jump_point = None
            for i in range(len(values)-1):
                if np.abs(values[i] - values[i+1]) > 2048:
                    if jump_point is not None:
                        raise ValueError(f"Multiple jumps found in calibration data: {values}")
                    jump_point = i

            if jump_point is None:
                as5600_preprocess = lambda x: x
            else:
                group_1, group_2 = values[:jump_point+1], values[jump_point+1:]
                max_1, min_1 = np.max(group_1), np.min(group_1)
                max_2, min_2 = np.max(group_2), np.min(group_2)

                if min_1 > max_2:
                    threshold = (min_1 + max_2) / 2
                elif min_2 > max_1:
                    threshold = (min_2 + max_1) / 2
                else:
                    raise ValueError(f"Overlap in two groups {values}")
                
                def as5600_preprocess(x):
                    if x < threshold:
                        return x + 4096
                    else:
                        return x

            values = list(map(as5600_preprocess, values))
            if values[0] > values[-1]:
                values = [values[0]+200] + values + [values[-1]-200]
            else:
                values = [values[0]-200] + values + [values[-1]+200]
            widths = list(widths)
            widths = [widths[0]] + widths + [widths[-1]]
            
            as5600_to_width = WrappedInterp1d(values, widths, extend_boundary=10)
            
        print(f"Processing data from {handcap_data_root} and {tracker_root}...")

        # --- Continuous Tracking Loaders ---
        tracker_full_path = os.path.join(args.data_root, tracker_root)

        # 1. HTC Pose Continuous Loader
        htc_pose_folder = os.path.join(tracker_full_path, "htc_pose")
        if not os.path.exists(htc_pose_folder):
            htc_pose_folder = tracker_full_path
            
        htc_pose_times = []
        htc_pose_values = []
        if os.path.exists(htc_pose_folder):
            pose_files = [f for f in os.listdir(htc_pose_folder) if f.endswith('.json')]
            pose_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))) if any(c.isdigit() for c in x) else x)
            for pose_file in pose_files:
                try:
                    with open(os.path.join(htc_pose_folder, pose_file), "r") as f:
                        pose_data = json.load(f)
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse {pose_file}: {e}")
                    continue
                ts_key = "timestamps" if "timestamps" in pose_data else "timestamps"
                data_key = "data" if "data" in pose_data else "data"
                htc_pose_times.extend(pose_data[ts_key])
                for p in pose_data[data_key]:
                    htc_pose_values.append(compute_pose_transform(p))
                        
        htc_pose_times = np.array(htc_pose_times)
        htc_pose_idx = 0

        print("No-phone mode: skipping phone_pose / phone_camera / phone_depth loaders.")

        # --- Handcap episodes logic ---
        handcap_full_path = os.path.join(args.data_root, handcap_data_root)
        try:
            episode_folders = [d for d in os.listdir(handcap_full_path) if os.path.isdir(os.path.join(handcap_full_path, d))]
        except Exception:
            episode_folders = []
            
        episode_folders.sort(key=lambda x: extract_episode_num(x))

        for cap in video_caps.values():
            cap.release()
        video_caps.clear()
        npz_cache = {}
        
        if batch_idx == 0:
            saved_idx = args.start_episode_idx
            
        for ep_folder in episode_folders:
            ep_root = os.path.join(handcap_full_path, ep_folder)
            
            ep_num = extract_episode_num(ep_folder)
            if ep_num < args.start_episode_idx:
                if batch_idx == 0:
                    print(f"Skipping episode {ep_folder} based on --start_episode_idx {args.start_episode_idx}.")
                continue
                
            print(
                "[AngleCalib][Episode] Using angle calibration map entry:\n"
                f"  saved_episode_idx={saved_idx}\n"
                f"  raw_episode={ep_folder}\n"
                f"  episode_root={ep_root}\n"
                f"  capture_id={capture_id}\n"
                f"  map_file={angle_calib_map_path}\n"
                f"  map_entry={capture_id}: {angle_calib_entry}\n"
                f"  resolved_angle_calib={angle_calib_path}"
            )
            print(f"Processing episode {saved_idx} from {ep_root}...")

            angle_dataroot = os.path.join(ep_root, "angle")
            force_dataroot = os.path.join(ep_root, "force")
            tactile_dataroot = os.path.join(ep_root, "tactile")
            wrist_dataroot = os.path.join(ep_root, "wrist")
            
            if not os.path.exists(wrist_dataroot):
                angle_dataroot = ep_root
                force_dataroot = ep_root
                tactile_dataroot = ep_root
                wrist_dataroot = ep_root

            try:
                angle_file_list = [f for f in os.listdir(angle_dataroot) if f.startswith("angle_") and f.endswith(".json")]
                left_tactile_file_list = [f for f in os.listdir(tactile_dataroot) if f.startswith("lefttactile_") and f.endswith(".json")]
                right_tactile_file_list = [f for f in os.listdir(tactile_dataroot) if f.startswith("righttactile_") and f.endswith(".json")]
                left_force_file_list = [f for f in os.listdir(force_dataroot) if f.startswith("leftforce_") and f.endswith(".json")]
                right_force_file_list = [f for f in os.listdir(force_dataroot) if f.startswith("rightforce_") and f.endswith(".json")]
                wrist_file_list = [f for f in os.listdir(wrist_dataroot) if f.startswith("wrist") and f.endswith(".json")]
            except Exception as e:
                print(f"Skipping {ep_root} due to missing structure. ({e})")
                continue

            # Load the modality data for this specific episode
            angle_times, angle_values = preload_angle_data(angle_dataroot, angle_file_list, "angle_", "angles", "timestamps")
            left_tactile_times, left_tactile_values = preload_video_data(tactile_dataroot, left_tactile_file_list, "lefttactile_")
            right_tactile_times, right_tactile_values = preload_video_data(tactile_dataroot, right_tactile_file_list, "righttactile_")
            wrist_times, wrist_values = preload_video_data(wrist_dataroot, wrist_file_list, "wrist_")
            left_force_times, left_force_values = preload_data(force_dataroot, left_force_file_list, "leftforce_", "forces", "timestamps")
            right_force_times, right_force_values = preload_data(force_dataroot, right_force_file_list, "rightforce_", "forces", "timestamps")

            def check_all_zero(force_list):
                if not force_list:
                    return True
                return np.all(np.array(force_list) == 0)

            left_zero = check_all_zero(left_force_values)
            right_zero = check_all_zero(right_force_values)

            if left_zero and not right_zero:
                print(f"Warning: left force is all zeros in {ep_root}, replacing with right force.")
                left_force_values = [v for v in right_force_values]
                left_force_times = right_force_times.copy()
            elif right_zero and not left_zero:
                print(f"Warning: right force is all zeros in {ep_root}, replacing with left force.")
                right_force_values = [v for v in left_force_values]
                right_force_times = left_force_times.copy()

            if len(wrist_times) == 0:
                print(f"No wrist data found in {ep_root}. Skipping...")
                continue
                
            if getattr(args, 'use_refer_tactile', False):
                left_calib_params, right_calib_params = None, None
            elif global_calib_params is not None:
                left_calib_params, right_calib_params = global_calib_params
                print(f"Using run-wide tactile calibration source: {global_calib_source}")
            else:
                raise RuntimeError("Run-wide tactile calibration parameters were not initialized.")

            angle_idx, ltactile_idx, rtactile_idx, lforce_idx, rforce_idx = 0, 0, 0, 0, 0
            
            frame_episode_idx = 0
            current_frame = None
            
            first_wrist_time = wrist_times[0]
            last_wrist_time = wrist_times[-1]
            start_pose, _ = find_closest_timestamp(first_wrist_time, htc_pose_times, htc_pose_values, 0, threshold=100.0)
            end_pose, _ = find_closest_timestamp(last_wrist_time, htc_pose_times, htc_pose_values, htc_pose_idx, threshold=100.0)
            
            if start_pose is None or end_pose is None:
                print(f"Warning: Missing continuous HTC pose for episode bounds. Skipping episode.")
                continue

            for time_idx, (wrist_video_path, wrist_frame_idx) in enumerate(wrist_values):
                each_time_step = wrist_times[time_idx]
                
                # Fetch aligned data across all available modalities. Phone data is absent in htc.py collections.
                current_pose, htc_pose_idx = find_closest_timestamp(each_time_step, htc_pose_times, htc_pose_values, htc_pose_idx)
                
                # Fetch aligned Handcap Episode components
                agree_value, angle_idx = find_closest_timestamp(each_time_step, angle_times, angle_values, angle_idx)
                left_tactile_dict, ltactile_idx = find_closest_timestamp(each_time_step, left_tactile_times, left_tactile_values, ltactile_idx)
                right_tactile_dict, rtactile_idx = find_closest_timestamp(each_time_step, right_tactile_times, right_tactile_values, rtactile_idx)
                left_force_value, lforce_idx = find_closest_timestamp(each_time_step, left_force_times, left_force_values, lforce_idx)
                right_force_value, rforce_idx = find_closest_timestamp(each_time_step, right_force_times, right_force_values, rforce_idx)
                
                required_data = [current_pose, agree_value, left_tactile_dict, right_tactile_dict, left_force_value, right_force_value]

                if any(x is None for x in required_data):
                    print(f"Skipping frame at time {each_time_step} due to missing aligned data.")
                    continue

                (left_tactile_video_path, left_tactile_frame_idx) = left_tactile_dict
                (right_tactile_video_path, right_tactile_frame_idx) = right_tactile_dict

                width_list = [[as5600_to_width(as5600_preprocess(x[0]))] for x in [agree_value]][0]
                
                try:
                    if getattr(args, 'use_refer_tactile', False):
                        left_tactile_value = np.zeros((224, 224, 3), dtype=np.uint8)
                        right_tactile_value = np.zeros((224, 224, 3), dtype=np.uint8)
                    else:
                        left_tactile_value = get_frame(left_tactile_video_path, left_tactile_frame_idx)
                        right_tactile_value = get_frame(right_tactile_video_path, right_tactile_frame_idx)
                        
                        left_tactile_value = apply_calibration(left_tactile_value, left_calib_params)
                        right_tactile_value = apply_calibration(right_tactile_value, right_calib_params)
                        left_tactile_value = resize_tactile_to_model_input(left_tactile_value)
                        right_tactile_value = resize_tactile_to_model_input(right_tactile_value)

                    wrist_value = resize_wrist_to_model_input(get_frame(wrist_video_path, wrist_frame_idx))
                except ValueError as e:
                    print(f"Warning: Skipping unreadable frame in {ep_root}: {e}")
                    continue

                state_concatenated = np.concatenate([
                    current_pose.astype(np.float32), 
                    np.array(width_list).astype(np.float32), 
                    np.array([0.0]).astype(np.float32), 
                    np.array([0.0]).astype(np.float32), 
                    np.array([0.0]).astype(np.float32)
                ])

                frame_data = {
                    "start_pose": start_pose.astype(np.float32),
                    "end_pose": end_pose.astype(np.float32),
                    "observation.state": state_concatenated,
                    "observation.images.wrist": wrist_value,
                    "observation.tactiles.left": left_tactile_value,
                    "observation.tactiles.right": right_tactile_value,
                    "observation.forces.left": np.array(left_force_value).astype(np.float32),
                    "observation.forces.right": np.array(right_force_value).astype(np.float32),
                    "task": args.task
                }

                if current_frame is None:
                    current_frame = frame_data
                else:
                    next_frame = frame_data
                    current_frame["action"] = next_frame["observation.state"]
                    
                    dataset.add_frame(current_frame)
                    print(f"Added frame {frame_episode_idx + 1} to episode {saved_idx}")
                    
                    frame_episode_idx += 1
                    current_frame = next_frame

            if frame_episode_idx > 0:
                dataset.save_episode(parallel_encoding=args.parallel_video_encoding)
                saved_idx += 1
                
            if saved_idx % 5 == 0:
                print(f"Releasing resources to free memory at episode {saved_idx}")
                for cap in video_caps.values():
                    cap.release()
                video_caps.clear()
                npz_cache.clear()
                import gc
                gc.collect()        
                
    for cap in video_caps.values():
        cap.release()

def main(args):
    dataset_repo = Path(args.output_root).name
    ensure_working_video_encoding(args)

    if os.path.exists(args.output_root):
        print(f"[WARNING!!!] Output folder {args.output_root} already exists. Whether to overwrite it? (y/n)")
        user_input = input("y/n:")
        user_input = user_input.strip().lower()
        if user_input != 'y':
            print("Exiting without overwriting.")
            sys.exit(0)
        else:
            print(f"Overwriting the existing folder {args.output_root}...")
            shutil.rmtree(args.output_root)
    
    dataset = LeRobotDataset.create(
            repo_id = dataset_repo,
            fps=24,
            root=args.output_root,
            use_videos=True,
            features=features,
            vcodec=args.vcodec,
            video_pix_fmt=args.video_pix_fmt,
            video_gop=args.video_gop,
            video_crf=args.video_crf,
            encoder_threads=args.encoder_threads,
        )
    
    with VideoEncodingManager(dataset):
        record_loop(dataset, args)

    print("Finalizing dataset buffers...")
    dataset.finalize()

    # ---------------------------------
    # Added for lingbot-va compatibility
    # ---------------------------------
    print("Injecting action_config for lingbot-va compatibility...")
    episodes_file = os.path.join(args.output_root, "meta", "episodes.jsonl")
    if os.path.exists(episodes_file):
        new_lines = []
        with open(episodes_file, 'r') as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)
                length = data.get("length", 0)
                # Assign default task text if not present
                if "tasks" not in data or not data["tasks"]:
                    data["tasks"] = ["perform the manipulation task"]
                
                # Assign single continuous action configuration
                data["action_config"] = [
                    {
                        "start_frame": 0,
                        "end_frame": length,
                        "action_text": data["tasks"][0],
                    }
                ]
                new_lines.append(json.dumps(data) + "\n")
        
        with open(episodes_file, 'w') as f:
            f.writelines(new_lines)
        print(f"Successfully updated {episodes_file} to lingbot-va format.")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--cap_id", type=str, default="all") 
    parser.add_argument("--data_root", type=str, default="handcap_data")
    parser.add_argument("--output_root", type=str, default="combined_data")
    parser.add_argument("--start_episode_idx", type=int, default=0)
    parser.add_argument("--task", type=str, default="peg_in_hole")
    parser.add_argument(
        "--vcodec",
        type=str,
        default="h264",
        choices=["auto", "h264", "libx264", "hevc", "libsvtav1", "h264_videotoolbox", "hevc_videotoolbox"],
        help="Video encoder. Default h264 avoids lower-quality hardware auto encoding.",
    )
    parser.add_argument(
        "--video_pix_fmt",
        type=str,
        default="yuv444p",
        choices=["yuv420p", "yuv444p"],
        help="Video pixel format. yuv444p preserves color detail better; use yuv420p if a decoder requires it.",
    )
    parser.add_argument(
        "--video_gop",
        type=int,
        default=2,
        help="Video keyframe interval. Smaller values improve frame seeking; 2 matches LeRobot defaults.",
    )
    parser.add_argument(
        "--video_crf",
        type=int,
        default=12,
        help="Video quality for CRF/QP encoders. Lower is higher quality; 12 is near visually lossless for H.264.",
    )
    parser.add_argument(
        "--encoder_threads",
        type=int,
        default=None,
        help="Threads per encoder. Useful for libsvtav1, e.g. --encoder_threads 1.",
    )
    parser.add_argument(
        "--parallel_video_encoding",
        action="store_true",
        help="Encode multiple camera videos in parallel. Faster, but can crash SVT-AV1 on memory/CPU pressure.",
    )
    parser.add_argument(
        "--use_refer_tactile",
        action="store_true",
        help="Use reference tactile images instead of video frames",
    )
    parser.add_argument(
        "--calibration_seed",
        type=int,
        default=None,
        help="Random seed for selecting the run-wide tactile calibration episode/frame. Default is random each run.",
    )
    parser.add_argument(
        "--angle_calib_map",
        type=str,
        required=True,
        help=(
            "JSON mapping from capture id (for example 51301) to the angle/gripper calibration file. "
            "Relative calibration paths are resolved from the handcap repo root, this script folder, "
            "and the map file folder."
        ),
    )

    args = parser.parse_args()
    main(args)
