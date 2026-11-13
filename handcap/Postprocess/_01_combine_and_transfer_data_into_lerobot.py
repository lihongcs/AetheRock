"""This module combines data from multiple times collected into a single dataset.
This is an INVERSE structure where Pose (and Phone Camera/Depth/Pose) are continuous, 
and Handcap is separated into episodes.
"""

import os
import cv2
import sys
import shutil
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lerobot30_env"))

import json
import numpy as np 
from pathlib import Path
from PIL import Image
from umi_utils.pose_util import *
from argparse import ArgumentParser
from scipy.interpolate import interp1d
from huggingface_hub.constants import HF_HOME
from tactile_calibration.calibrate_tactile import compute_calibration_params, apply_calibration
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.datasets.image_writer import safe_stop_image_writer
from lerobot.datasets.dataset_metadata import CODEBASE_VERSION as LEROBOT_CODEBASE_VERSION
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata, LeRobotDataset

if LEROBOT_CODEBASE_VERSION != "v3.0":
    raise RuntimeError(f"FATAL ERROR: The imported LeRobot engine is currently at version {LEROBOT_CODEBASE_VERSION}, but v3.0 is STRICTLY REQUIRED! Please check your local lerobot30 copies.")

from scipy.spatial.transform import Rotation



########################################### interpolator ###########################################

def resize_with_black_padding(image, target_h=480, target_w=640, is_depth=False):
    """
    Resize image to fit within (target_h, target_w) while preserving aspect ratio,
    then pad with black borders to reach exactly (target_h, target_w).
    Uses INTER_NEAREST for depth maps to avoid float interpolation artifacts.
    """
    h, w = image.shape[:2]
    c = image.shape[2] if len(image.shape) == 3 else 1
        
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    interp = cv2.INTER_NEAREST if is_depth else cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_w, new_h), interpolation=interp)
    
    if len(resized.shape) == 2:
        resized = np.expand_dims(resized, axis=-1)
        
    pad_w = (target_w - new_w) // 2
    pad_h = (target_h - new_h) // 2
    
    if c == 3:
        canvas = np.zeros((target_h, target_w, 3), dtype=image.dtype)
    else:
        canvas = np.zeros((target_h, target_w, 1), dtype=image.dtype)
        
    canvas[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
    return canvas


def resize_tactile_to_model_input(image, target_size=(224, 224)):
    """Keep tactile data geometry aligned with collection and model input."""
    target_w, target_h = target_size
    if image.shape[:2] == (target_h, target_w):
        return image
    return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)


def resize_wrist_to_model_input(image, target_size=(224, 224)):
    """Match openpi_client.image_tools.resize_with_pad for real-robot wrist images."""
    target_w, target_h = target_size
    if image.shape[:2] == (target_h, target_w):
        return image

    pil_image = Image.fromarray(image)
    cur_width, cur_height = pil_image.size
    ratio = max(cur_width / target_w, cur_height / target_h)
    resized_width = int(cur_width / ratio)
    resized_height = int(cur_height / ratio)
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    resized = pil_image.resize((resized_width, resized_height), resample=resampling)

    canvas = Image.new(resized.mode, (target_w, target_h), 0)
    pad_width = max(0, int((target_w - resized_width) / 2))
    pad_height = max(0, int((target_h - resized_height) / 2))
    canvas.paste(resized, (pad_width, pad_height))
    return np.asarray(canvas, dtype=image.dtype)


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

def compute_phone_pose_transform(raw_pose):
    raw_pose = np.array(raw_pose)
    pose_mat = pose_to_mat(quat_to_rot(raw_pose))
    return mat_to_certain_pose_type(pose_mat, pose_type="rotvec")



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
        "observation.state_phone": {
            "dtype": "float32",
            "shape": (6,),
            "names": ["x", "y", "z", "qx", "qy", "qz"]
        },
        "observation.images.wrist": {
            "dtype": "video",
            "shape": (224, 224, 3),
            "names": ["height", "width", "channels"],
            "info": None
        },
        "observation.images.phone": {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channels"],
            "info": None
        },
        "observation.images.phone_depth": {
            "dtype": "image",
            "shape": (480, 640, 1),
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
    tracker_file_list = []
    if args.cap_id.lower() == "all":
        tracker_file_list = [f for f in file_list if f.startswith("htc")]
    else:
        cap_ids = args.cap_id.split(" ")
        for tf in cap_ids:
            tracker_file_list += [f for f in file_list if f.startswith("htc") and tf in f]
        
    tracker_file_list = sorted(list(set(tracker_file_list)))
    
    if not tracker_file_list:
        print(f"Error: No matching data folders found in {args.data_root} for cap_ids {cap_ids}.")
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
                timestamps = timestamps['timestamps'] # For new phone chunk formats

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
    
    for batch_idx, (handcap_data_root, tracker_root) in enumerate(zip(handcap_file_list, tracker_file_list)):
        
        angle_calib_path = "../Preprocess/angle_calib_params/" + "gripper_calibration_interpolation50501.json"
        with open(angle_calib_path, "r") as fp:
            width_calib = json.load(fp)["calibration"]
            widths, values = zip(*width_calib)
            
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

        # 2. Phone Pose Continuous Loader
        phone_pose_folder = os.path.join(tracker_full_path, "phone_pose")
        phone_pose_times = []
        phone_pose_values = []
        if os.path.exists(phone_pose_folder):
            ph_pose_files = [f for f in os.listdir(phone_pose_folder) if f.endswith('.json')]
            ph_pose_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))) if any(c.isdigit() for c in x) else x)
            for ph_file in ph_pose_files:
                try:
                    with open(os.path.join(phone_pose_folder, ph_file), "r") as f:
                        ph_data = json.load(f)
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse {ph_file}: {e}")
                    continue
                phone_pose_times.extend(ph_data.get("timestamps", []))
                for p in ph_data.get("data", []):
                    # Phone pose doesn't need external transform matrix mapping
                    phone_pose_values.append(compute_phone_pose_transform(p))
        phone_pose_times = np.array(phone_pose_times)
        phone_pose_idx = 0

        # 3. Phone Camera Multi-Loader (lazy-loading .mp4)
        phone_cam_folder = os.path.join(tracker_full_path, "phone_camera")
        phone_cam_times = np.array([])
        phone_cam_values = []
        if os.path.exists(phone_cam_folder):
            cam_jsons = [f for f in os.listdir(phone_cam_folder) if f.endswith('.json')]
            phone_cam_times, phone_cam_values = preload_video_data(phone_cam_folder, cam_jsons, "phone_camera_chunk_")
        phone_cam_idx = 0

        # 4. Phone Depth Multi-Loader (lazy-loading .npz)
        phone_depth_folder = os.path.join(tracker_full_path, "phone_depth")
        phone_depth_times = np.array([])
        phone_depth_values = []
        if os.path.exists(phone_depth_folder):
            depth_jsons = [f for f in os.listdir(phone_depth_folder) if f.endswith('.json')]
            # Assume depth arrays are saved as matching .npz files for each .json chunk
            phone_depth_times, phone_depth_values = preload_video_data(phone_depth_folder, depth_jsons, "depth_chunk_", mp4_ext=".npz")
        phone_depth_idx = 0

        # --- Handcap episodes logic ---
        handcap_full_path = os.path.join(args.data_root, handcap_data_root)
        try:
            episode_folders = [d for d in os.listdir(handcap_full_path) if os.path.isdir(os.path.join(handcap_full_path, d))]
        except Exception:
            episode_folders = []
            
        def extract_episode_num(name):
            nums = ''.join(filter(str.isdigit, name))
            return int(nums) if nums else 999999
            
        episode_folders.sort(key=lambda x: extract_episode_num(x))

        video_caps = {} 
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

            def get_npz_frame(npz_path, frame_idx):
                if npz_path not in npz_cache:
                    npz_cache.clear() # Cache limits memory to essentially just 1 chunk at a time
                    data = np.load(npz_path)
                    key = "depths" if "depths" in data else list(data.keys())[0]
                    npz_cache[npz_path] = data[key]
                    
                if frame_idx >= len(npz_cache[npz_path]):
                    raise ValueError(f"Failed to read frame {frame_idx} from {npz_path}: index out of range.")
                arr = npz_cache[npz_path][frame_idx]
                return arr

            if global_calib_params is None:
                calib_saved_path = os.path.join(args.data_root, "tactile_calib.json")
                if len(left_tactile_values) > 0 and len(right_tactile_values) > 0:
                    try:
                        l_vid, l_idx = left_tactile_values[0]
                        left_calib_params = compute_calibration_params(cv2.cvtColor(get_frame(l_vid, l_idx), cv2.COLOR_RGB2BGR))
                        
                        r_vid, r_idx = right_tactile_values[0]
                        right_calib_params = compute_calibration_params(cv2.cvtColor(get_frame(r_vid, r_idx), cv2.COLOR_RGB2BGR))

                        calib_data = {
                            "left_H": left_calib_params[0].tolist(), "left_mask": left_calib_params[1].tolist(), "left_w": int(left_calib_params[2]), "left_h": int(left_calib_params[3]),
                            "right_H": right_calib_params[0].tolist(), "right_mask": right_calib_params[1].tolist(), "right_w": int(right_calib_params[2]), "right_h": int(right_calib_params[3])
                        }
                        with open(calib_saved_path, 'w') as f:
                            json.dump(calib_data, f)
                        
                        global_calib_params = (left_calib_params, right_calib_params)
                        print(f"Calibration parameters confirmed and saved to {calib_saved_path}")
                    except ValueError as e:
                        print(f"Warning: Tactile calibration frame unavailable in {ep_root}: {e}")

            if global_calib_params is not None:
                left_calib_params, right_calib_params = global_calib_params
            else:
                left_calib_params, right_calib_params = None, None

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
                
                # Fetch aligned data across all continuous extra modalities
                current_pose, htc_pose_idx = find_closest_timestamp(each_time_step, htc_pose_times, htc_pose_values, htc_pose_idx)
                curr_phone_pose, phone_pose_idx = find_closest_timestamp(each_time_step, phone_pose_times, phone_pose_values, phone_pose_idx)
                ph_cam_dict, phone_cam_idx = find_closest_timestamp(each_time_step, phone_cam_times, phone_cam_values, phone_cam_idx)
                ph_depth_dict, phone_depth_idx = find_closest_timestamp(each_time_step, phone_depth_times, phone_depth_values, phone_depth_idx)
                
                # Fetch aligned Handcap Episode components
                agree_value, angle_idx = find_closest_timestamp(each_time_step, angle_times, angle_values, angle_idx)
                left_tactile_dict, ltactile_idx = find_closest_timestamp(each_time_step, left_tactile_times, left_tactile_values, ltactile_idx)
                right_tactile_dict, rtactile_idx = find_closest_timestamp(each_time_step, right_tactile_times, right_tactile_values, rtactile_idx)
                left_force_value, lforce_idx = find_closest_timestamp(each_time_step, left_force_times, left_force_values, lforce_idx)
                right_force_value, rforce_idx = find_closest_timestamp(each_time_step, right_force_times, right_force_values, rforce_idx)
                
                required_data = [current_pose, curr_phone_pose, ph_cam_dict, ph_depth_dict, agree_value, left_tactile_dict, right_tactile_dict, left_force_value, right_force_value]

                if any(x is None for x in required_data):
                    print(f"Skipping frame at time {each_time_step} due to missing aligned data.")
                    continue

                (left_tactile_video_path, left_tactile_frame_idx) = left_tactile_dict
                (right_tactile_video_path, right_tactile_frame_idx) = right_tactile_dict
                (ph_cam_path, ph_cam_frame_idx) = ph_cam_dict
                (ph_depth_path, ph_depth_frame_idx) = ph_depth_dict

                width_list = [[as5600_to_width(as5600_preprocess(x[0]))] for x in [agree_value]][0]
                
                try:
                    if getattr(args, 'use_refer_tactile', False):
                        left_tactile_value = np.zeros((224, 224, 3), dtype=np.uint8)
                        right_tactile_value = np.zeros((224, 224, 3), dtype=np.uint8)
                    else:
                        left_tactile_value = get_frame(left_tactile_video_path, left_tactile_frame_idx)
                        right_tactile_value = get_frame(right_tactile_video_path, right_tactile_frame_idx)
                        
                        # tactile calibration 
                        if left_calib_params is not None:
                            left_tactile_value = apply_calibration(left_tactile_value, left_calib_params)
                        if right_calib_params is not None:
                            right_tactile_value = apply_calibration(right_tactile_value, right_calib_params)
                        left_tactile_value = resize_tactile_to_model_input(left_tactile_value)
                        right_tactile_value = resize_tactile_to_model_input(right_tactile_value)

                    wrist_value = resize_wrist_to_model_input(get_frame(wrist_video_path, wrist_frame_idx))
                    phone_value = resize_with_black_padding(get_frame(ph_cam_path, ph_cam_frame_idx))
                    
                    # Retrieve and pad phone depth
                    depth_raw = get_npz_frame(ph_depth_path, ph_depth_frame_idx)
                    phone_depth_value = resize_with_black_padding(depth_raw, is_depth=True).astype(np.float32)
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

                curr_phone_pose = curr_phone_pose.astype(np.float32)

                frame_data = {
                    "start_pose": start_pose.astype(np.float32),
                    "end_pose": end_pose.astype(np.float32),
                    "observation.state": state_concatenated,
                    "observation.state_phone": curr_phone_pose,
                    "observation.images.wrist": wrist_value,
                    "observation.images.phone": phone_value,
                    "observation.images.phone_depth": phone_depth_value,
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
        default="auto",
        choices=["auto", "h264", "hevc", "libsvtav1", "h264_videotoolbox", "hevc_videotoolbox"],
        help="Video encoder. 'auto' prefers hardware encoders such as h264_videotoolbox on macOS.",
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

    args = parser.parse_args()
    main(args)
