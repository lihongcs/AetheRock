"""This module combines Handcap raw sensor data and turns it into NVIDIA Cosmos Policy ALOHA HDF5 format."""

import sys
import os
import shutil
import cv2
import json
import h5py
import random
import numpy as np 
from pathlib import Path
from argparse import ArgumentParser
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation

# Import your custom umi features from local
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lerobot30_env"))
from umi_utils.pose_util import pose_to_mat, quat_to_rot, mat_to_certain_pose_type
from tactile_calibration.calibrate_tactile import compute_calibration_params, apply_calibration

########################################### interpolator & utils ###########################################

def resize_with_black_padding(image, target_h=480, target_w=640):
    """
    Resize image to fit within (target_h, target_w) while preserving aspect ratio,
    then pad with black borders to reach exactly (target_h, target_w).
    """
    h, w = image.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(image, (new_w, new_h))
    pad_w = (target_w - new_w) // 2
    pad_h = (target_h - new_h) // 2
    canvas = np.zeros((target_h, target_w, 3), dtype=image.dtype)
    canvas[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
    return canvas

class WrappedInterp1d:
    def __init__(self, X, Y, name="interpolator"):
        sort_id = np.argsort(X)
        X = np.array(X)[sort_id]
        Y = [Y[i] for i in sort_id]
        self.traj_interp = interp1d(X, Y, axis=0, bounds_error=True)
        self.name = name
        self.x_min = X[0]
        self.x_max = X[-1]
        self.left_val = Y[0]
        self.right_val = Y[-1]
        self._warned_left = False
        self._warned_right = False

    def __call__(self, x):
        if x <= self.x_min:
            if x < self.x_min and not self._warned_left:
                print(
                    f"[WARNING] {self.name}: value {x} is below calibration "
                    f"range min {self.x_min}; clamping to {self.left_val}.")
                self._warned_left = True
            return self.left_val
        elif x >= self.x_max:
            if x > self.x_max and not self._warned_right:
                print(
                    f"[WARNING] {self.name}: value {x} is above calibration "
                    f"range max {self.x_max}; clamping to {self.right_val}.")
                self._warned_right = True
            return self.right_val
        else:
            return self.traj_interp(x)

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

def create_video_from_images(images, video_path, fps=25):
    """Creates an MP4 video from a sequence of images."""
    if len(images) == 0: return
    height, width, channels = images[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    for img in images:
        if channels == 3:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = img
        out.write(img_bgr)
    out.release()

def pad_to_aloha_dim(state_array, target_dim=14):
    """Pads N-dimensional state with zeros to reach target_dim (ALOHA default 14)"""
    pad_len = target_dim - len(state_array)
    if pad_len > 0:
        return np.concatenate([state_array, np.zeros(pad_len, dtype=np.float32)])
    return state_array[:target_dim].astype(np.float32)

########################################### main execution ###########################################

def record_loop(args, start_idx=0):
    
    file_list = os.listdir(args.data_root)
    tracker_file_list = []
    if args.cap_id.lower() == "all":
        tracker_file_list = [f for f in file_list if f.startswith("htc")]
    else:
        cap_ids = args.cap_id.split(" ")
        for tf in cap_ids:
            tracker_file_list += [f for f in file_list if f.startswith("htc") and tf in f]
        
    tracker_file_list = set(sorted(tracker_file_list))
    handcap_file_list = ["handcap_" + f.split("_")[-1] for f in tracker_file_list]
    
    def preload_data(dataroot, file_list, prefix, value_key, time_key, is_npz=False):
        all_values, all_times = [], []
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
                    print(f"[WARNING] Skipping malformed JSON {file_path}: {e}")
                    continue
                if value_key not in data or time_key not in data:
                    print(
                        f"[WARNING] Skipping JSON with missing keys {file_path}: "
                        f"expected {value_key!r} and {time_key!r}")
                    continue
                all_values.extend(data[value_key])
                all_times.extend(data[time_key])
        return np.array(all_times), all_values

    def preload_video_data(dataroot, file_list, prefix):
        all_times, all_video_paths, all_frame_indices = [], [], []
        for file_name in sorted(file_list, key=lambda f: int(f.split('_')[-1].split('.')[0]) if '_' in f else 0):
            json_path = os.path.join(dataroot, file_name)
            video_path = os.path.join(dataroot, file_name.replace('.json', '.mp4'))
            try:
                with open(json_path, 'r') as f:
                    timestamps = json.load(f)
            except json.JSONDecodeError as e:
                print(f"[WARNING] Skipping malformed video timestamp JSON {json_path}: {e}")
                continue
            if not os.path.exists(video_path):
                print(f"[WARNING] Missing video for {json_path}. Skipping this chunk.")
                continue

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"[WARNING] Could not open {video_path}. Skipping this chunk.")
                continue
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            valid_len = min(len(timestamps), frame_count)
            if valid_len <= 0:
                print(f"[WARNING] No readable frames in {video_path}. Skipping this chunk.")
                continue
            if valid_len < len(timestamps):
                print(
                    f"[WARNING] Truncating {json_path}: "
                    f"{len(timestamps)} timestamps but {frame_count} video frames."
                )

            all_times.extend(timestamps[:valid_len])
            all_video_paths.extend([video_path] * valid_len)
            all_frame_indices.extend(range(valid_len))
        return np.array(all_times), list(zip(all_video_paths, all_frame_indices))
    
    global_calib_params = None
    all_episodes_data = [] # To hold RAM representations before saving Train/Val split

    for batch_idx, (handcap_data_root, tracker_root) in enumerate(zip(handcap_file_list, tracker_file_list)):
        angle_calib_path = "../Preprocess/angle_calib_params/" + "gripper_calibration_interpolation50501.json"
        with open(angle_calib_path, "r") as fp:
            width_calib = json.load(fp)["calibration"]
            widths, values = zip(*width_calib)
            
            jump_point = None
            for i in range(len(values)-1):
                if np.abs(values[i] - values[i+1]) > 2048:
                    jump_point = i

            if jump_point is None:
                as5600_preprocess = lambda x: x
            else:
                group_1, group_2 = values[:jump_point+1], values[jump_point+1:]
                max_1, min_1 = np.max(group_1), np.min(group_1)
                max_2, min_2 = np.max(group_2), np.min(group_2)
                threshold = (min_1 + max_2) / 2 if min_1 > max_2 else (min_2 + max_1) / 2
                def as5600_preprocess(x): return x + 4096 if x < threshold else x

            values = list(map(as5600_preprocess, values))
            values = [values[0] + (200 if values[0] < values[-1] else -200)] + values + [values[-1] + (-200 if values[0] < values[-1] else 200)]
            widths = [widths[0]] + list(widths) + [widths[-1]]
            as5600_to_width = WrappedInterp1d(values, widths, name="AS5600 angle->width")
            
        print(f"Loading continuous tracker data from {tracker_root}...")
        tracker_full_path = os.path.join(args.data_root, tracker_root)
        htc_pose_folder = os.path.join(tracker_full_path, "htc_pose")
        if not os.path.exists(htc_pose_folder):
             htc_pose_folder = tracker_full_path
             
        pose_files = [f for f in os.listdir(htc_pose_folder) if f.endswith('.json')]
        pose_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))) if any(c.isdigit() for c in x) else x)

        continuous_pose_times, continuous_pose_values = [], []
        for pose_file in pose_files:
            with open(os.path.join(htc_pose_folder, pose_file), "r") as f:
                pose_data = json.load(f)
                continuous_pose_times.extend(pose_data["timestamps"])
                for p in pose_data["data"]:
                    continuous_pose_values.append(compute_pose_transform(p))
                    
        continuous_pose_times = np.array(continuous_pose_times)
        continuous_pose_idx = 0

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
        if batch_idx == 0: saved_idx = start_idx
            
        for ep_folder in episode_folders:
            ep_root = os.path.join(handcap_full_path, ep_folder)
            print(f"Extracting memory chunk for Cosmos -> Episode {saved_idx} ({ep_root})")

            angle_dataroot = os.path.join(ep_root, "angle")
            force_dataroot = os.path.join(ep_root, "force")
            tactile_dataroot = os.path.join(ep_root, "tactile")
            wrist_dataroot = os.path.join(ep_root, "wrist")
            if not os.path.exists(wrist_dataroot):
                angle_dataroot = force_dataroot = tactile_dataroot = wrist_dataroot = ep_root

            try:
                angle_file_list = [f for f in os.listdir(angle_dataroot) if f.startswith("angle_") and f.endswith(".json")]
                lt_list = [f for f in os.listdir(tactile_dataroot) if f.startswith("lefttactile_") and f.endswith(".json")]
                rt_list = [f for f in os.listdir(tactile_dataroot) if f.startswith("righttactile_") and f.endswith(".json")]
                lf_list = [f for f in os.listdir(force_dataroot) if f.startswith("leftforce_") and f.endswith(".json")]
                rf_list = [f for f in os.listdir(force_dataroot) if f.startswith("rightforce_") and f.endswith(".json")]
                wrist_file_list = [f for f in os.listdir(wrist_dataroot) if f.startswith("wrist") and f.endswith(".json")]
            except Exception as e:
                continue

            angle_times, angle_values = preload_data(angle_dataroot, angle_file_list, "angle_", "angles", "timestamps")
            left_tactile_times, left_tactile_values = preload_video_data(tactile_dataroot, lt_list, "lefttactile_")
            right_tactile_times, right_tactile_values = preload_video_data(tactile_dataroot, rt_list, "righttactile_")
            wrist_times, wrist_values = preload_video_data(wrist_dataroot, wrist_file_list, "wrist")
            left_force_times, left_force_values = preload_data(force_dataroot, lf_list, "leftforce_", "forces", "timestamps")
            right_force_times, right_force_values = preload_data(force_dataroot, rf_list, "rightforce_", "forces", "timestamps")

            if len(wrist_times) == 0: continue
                
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
                    raise ValueError(f"Failed to read frame {frame_idx} from {video_path}")
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if global_calib_params is None:
                if len(left_tactile_values) > 0 and len(right_tactile_values) > 0:
                    try:
                        l_vid, l_idx = left_tactile_values[0]
                        left_c = compute_calibration_params(cv2.cvtColor(get_frame(l_vid, l_idx), cv2.COLOR_RGB2BGR))
                        r_vid, r_idx = right_tactile_values[0]
                        right_c = compute_calibration_params(cv2.cvtColor(get_frame(r_vid, r_idx), cv2.COLOR_RGB2BGR))
                        global_calib_params = (left_c, right_c)
                    except ValueError as e:
                        print(f"[WARNING] Tactile calibration frame unavailable in {ep_root}: {e}")

            if global_calib_params is not None:
                left_calib_params, right_calib_params = global_calib_params
            else:
                left_calib_params, right_calib_params = None, None

            angle_idx = ltactile_idx = rtactile_idx = lforce_idx = rforce_idx = 0
            
            # Extract buffers
            ep_qpos = []
            ep_action = []  
            ep_wrist_imgs = []
            ep_ltactile_imgs = []
            ep_rtactile_imgs = []
            
            first_wrist_time = wrist_times[0]
            last_wrist_time = wrist_times[-1]
            start_pose, _ = find_closest_timestamp(first_wrist_time, continuous_pose_times, continuous_pose_values, 0, threshold=100.0)
            end_pose, _ = find_closest_timestamp(last_wrist_time, continuous_pose_times, continuous_pose_values, continuous_pose_idx, threshold=100.0)
            
            if start_pose is None or end_pose is None: continue

            for time_idx, (wrist_video_path, wrist_frame_idx) in enumerate(wrist_values):
                each_time_step = wrist_times[time_idx]
                
                current_pose, continuous_pose_idx = find_closest_timestamp(each_time_step, continuous_pose_times, continuous_pose_values, continuous_pose_idx)
                agree_value, angle_idx = find_closest_timestamp(each_time_step, angle_times, angle_values, angle_idx)
                left_tactile_dict, ltactile_idx = find_closest_timestamp(each_time_step, left_tactile_times, left_tactile_values, ltactile_idx)
                right_tactile_dict, rtactile_idx = find_closest_timestamp(each_time_step, right_tactile_times, right_tactile_values, rtactile_idx)
                left_force_value, lforce_idx = find_closest_timestamp(each_time_step, left_force_times, left_force_values, lforce_idx)
                right_force_value, rforce_idx = find_closest_timestamp(each_time_step, right_force_times, right_force_values, rforce_idx)
                
                if not all([current_pose is not None, agree_value, left_tactile_dict, right_tactile_dict, left_force_value, right_force_value]):
                    continue

                (left_tactile_video_path, left_tactile_frame_idx) = left_tactile_dict
                (right_tactile_video_path, right_tactile_frame_idx) = right_tactile_dict

                width_list = [[as5600_to_width(as5600_preprocess(x[0]))] for x in [agree_value]][0]
                
                try:
                    left_tactile_value = get_frame(left_tactile_video_path, left_tactile_frame_idx)
                    right_tactile_value = get_frame(right_tactile_video_path, right_tactile_frame_idx)
                    wrist_value = get_frame(wrist_video_path, wrist_frame_idx)
                except ValueError as e:
                    print(f"[WARNING] Skipping unreadable frame in {ep_root}: {e}")
                    continue
                
                if left_calib_params is not None: left_tactile_value = apply_calibration(left_tactile_value, left_calib_params)
                if right_calib_params is not None: right_tactile_value = apply_calibration(right_tactile_value, right_calib_params)
                
                wrist_value = resize_with_black_padding(wrist_value, 480, 640)
                
                # BASE 10-DOF STATE: 6D Pose + 1D Width + 3 Zeros
                state_10d = np.concatenate([current_pose.astype(np.float32), np.array(width_list).astype(np.float32), np.array([0.0, 0.0, 0.0]).astype(np.float32)])
                # Pad to ALOHA 14-DOF Target
                padded_14d_state = pad_to_aloha_dim(state_10d, 14)

                ep_qpos.append(padded_14d_state)
                ep_wrist_imgs.append(wrist_value)
                ep_ltactile_imgs.append(left_tactile_value)
                ep_rtactile_imgs.append(right_tactile_value)

            if len(ep_qpos) > 1:
                ep_qpos = np.array(ep_qpos)
                
                # In LeRobot/ALOHA: action at step t = intended qpos at step t+1
                ep_action = np.zeros_like(ep_qpos)
                ep_action[:-1] = ep_qpos[1:]
                ep_action[-1] = ep_qpos[-1] # repeat last action
                
                all_episodes_data.append({
                    "original_id": saved_idx,
                    "task": getattr(args, "task", ""),
                    "qpos": ep_qpos,
                    "action": ep_action,
                    "images": {
                        "cam_high": np.stack(ep_wrist_imgs),
                        "cam_left_wrist": np.stack(ep_ltactile_imgs),
                        "cam_right_wrist": np.stack(ep_rtactile_imgs)
                    }
                })
                saved_idx += 1
                
            if saved_idx % 5 == 0:
                for cap in video_caps.values(): cap.release()
                video_caps.clear()
                import gc; gc.collect()        
                
    for cap in video_caps.values(): cap.release()
    
    return all_episodes_data


def save_cosmos_hdf5(out_dataset_dir, data_dict, episode_idx):
    """Saves HDF5 episode exactly imitating NVIDIA Cosmos ALOHA architecture."""
    out_path = os.path.join(out_dataset_dir, f"episode_{episode_idx}.hdf5")
    
    with h5py.File(out_path, "w", rdcc_nbytes=1024**2 * 2) as root:
        episode_len = data_dict["qpos"].shape[0]
        root.attrs["sim"] = False # Handcap is Real World
        if data_dict.get("task"):
            root.attrs["task"] = data_dict["task"]

        obs = root.create_group("observations")
        _ = obs.create_dataset("qpos", (episode_len, 14))
        _ = obs.create_dataset("qvel", (episode_len, 14)) # Cosmos ALOHA dummy zeroes
        _ = obs.create_dataset("effort", (episode_len, 14)) # Cosmos ALOHA dummy zeroes
        
        root["/observations/qpos"][...] = data_dict["qpos"]
        root["/observations/qvel"][...] = np.zeros((episode_len, 14), dtype=np.float32)
        root["/observations/effort"][...] = np.zeros((episode_len, 14), dtype=np.float32)

        # Video paths strings mapped for Cosmos Dataloader
        video_paths_group = obs.create_group("video_paths")
        for cam_name, relative_path in data_dict["video_paths"].items():
            video_paths_group.create_dataset(cam_name, data=relative_path.encode("utf-8"))

        _ = root.create_dataset("action", (episode_len, 14))
        root["/action"][...] = data_dict["action"]
        
        # relative_action is requested by Cosmos
        relative_actions = np.zeros_like(data_dict["action"])
        relative_actions[:-1] = data_dict["action"][1:] - data_dict["action"][:-1] 
        relative_actions[-1] = relative_actions[-2]
        _ = root.create_dataset("relative_action", (episode_len, 14))
        root["/relative_action"][...] = relative_actions

    print(f"✅ Saved Cosmos formatted HDF5 chunk: {out_path}")

def dump_train_val(all_episodes, output_root, val_percent=0.05, fps=24, existing_metadata=None):
    """Splits into Train/Val and dumps HDF5 + compressed MP4 chunks."""
    indices = list(range(len(all_episodes)))
    random.shuffle(indices)
    
    num_val = int(len(all_episodes) * val_percent)
    train_indices = indices[num_val:]
    val_indices = indices[:num_val]
    
    train_dir = os.path.join(output_root, "train")
    val_dir = os.path.join(output_root, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    metadata_list = []
    train_offset = 0
    val_offset = 0
    total_offset = 0
    
    if existing_metadata:
        train_offset = existing_metadata.get("dataset_info", {}).get("train_episodes", 0)
        val_offset = existing_metadata.get("dataset_info", {}).get("val_episodes", 0)
        total_offset = existing_metadata.get("dataset_info", {}).get("total_episodes", 0)
        metadata_list = existing_metadata.get("episode_mapping", [])
    
    def _export_subset(idx_list, split_dest, split_name, offset):
        for i, orig_idx in enumerate(idx_list):
            running_idx = offset + i
            ep_data = all_episodes[orig_idx]
            video_paths = {}
            for cam_key, img_stack in ep_data["images"].items():
                v_filename = f"episode_{running_idx}_{cam_key}.mp4"
                v_fullpath = os.path.join(split_dest, v_filename)
                create_video_from_images(img_stack, v_fullpath, fps=fps)
                video_paths[cam_key] = v_filename
                
            ep_data["video_paths"] = video_paths
            save_cosmos_hdf5(split_dest, ep_data, running_idx)
            
            metadata_list.append({
                "original_file_name": f"handcap_episode_{ep_data['original_id']}",
                "preprocessed_episode_name": f"episode_{running_idx}.hdf5",
                "preprocessed_video_files": list(video_paths.values()),
                "split": split_name,
                "preprocessed_index": running_idx,
                "episode_length": ep_data["qpos"].shape[0],
                "is_simulation": False,
                "task": ep_data.get("task", ""),
            })
            
    print(f"\n[Cosmos Generation] Beginning TRAIN set packaging ({len(train_indices)} episodes)...")
    _export_subset(train_indices, train_dir, "train", train_offset)
    
    print(f"\n[Cosmos Generation] Beginning VAL set packaging ({len(val_indices)} episodes)...")
    _export_subset(val_indices, val_dir, "val", val_offset)
    
    return metadata_list, total_offset + len(indices), train_offset + len(train_indices), val_offset + len(val_indices)

def main(args):
    existing_metadata = None
    if os.path.exists(args.output_root):
        meta_path = os.path.join(args.output_root, "preprocessing_metadata.json")
        if os.path.exists(meta_path):
            print(f"[INFO] Found existing dataset at {args.output_root}. Resuming and appending to it.")
            with open(meta_path, "r") as f:
                existing_metadata = json.load(f)
        else:
            print(f"[WARNING] Output folder {args.output_root} exists but no metadata found. Overwriting...")
            shutil.rmtree(args.output_root)
            os.makedirs(args.output_root, exist_ok=True)
    else:
        os.makedirs(args.output_root, exist_ok=True)
        
    start_idx = args.start_episode_idx
    if existing_metadata and "episode_mapping" in existing_metadata:
        try:
            max_orig = max([int(m["original_file_name"].split("_")[-1]) for m in existing_metadata["episode_mapping"] if "original_file_name" in m and m["original_file_name"].split("_")[-1].isdigit()] + [0])
            start_idx = max(start_idx, max_orig + 1)
        except Exception:
            pass

    # Process episodes dynamically
    print(">> Phase 1: Mining and interpolating generic hardware episodes to RAM...")
    all_episodes = record_loop(args, start_idx)
    if not all_episodes:
        print("ERROR: No valid episodes parsed.")
        return
        
    print(f">> Phase 2: Compiling to Cosmos Policy Native HDF5 + MP4 Layouts...")
    metadata, t_count, tr_count, v_count = dump_train_val(all_episodes, args.output_root, val_percent=args.percent_val, existing_metadata=existing_metadata)
    
    all_metadata = {
        "dataset_info": {
            "original_dataset_path": args.data_root,
            "preprocessed_dataset_path": args.output_root,
            "total_episodes": t_count,
            "train_episodes": tr_count,
            "val_episodes": v_count,
            "validation_percentage": args.percent_val,
            "task": args.task,
            "preprocessing_settings": {"video_fps": 24},
        },
        "episode_mapping": metadata,
    }

    metadata_path = os.path.join(args.output_root, "preprocessing_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(all_metadata, f, indent=2)

    print("\n=========================================================")
    print("🎉 Cosmos-Policy Dataset Cross-Compilation Complete! ")
    print(f"-> Metadata generated: {metadata_path}")
    print(f"-> You can now point BASE_DATASETS_DIR to: {args.output_root}")
    print("=========================================================")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--cap_id", type=str, default="all")
    parser.add_argument("--data_root", type=str, default="handcap_data")
    parser.add_argument("--output_root", type=str, default="cosmos_ready_data")
    parser.add_argument("--start_episode_idx", type=int, default=0)
    parser.add_argument("--percent_val", type=float, default=0.01)
    parser.add_argument("--task", type=str, default="")

    args = parser.parse_args()
    main(args)
