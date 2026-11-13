import os
import cv2
import sys
import shutil
import json
import numpy as np 
from pathlib import Path
from argparse import ArgumentParser
from scipy.interpolate import interp1d
import scipy.spatial.transform as st
import zarr
from collections import defaultdict
import concurrent.futures
from tqdm import tqdm

# Adjust path to find modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from umi_utils.pose_util import *
from dp_utils.replay_buffer import ReplayBuffer
from dp_utils.imagecodecs_numcodecs import register_codecs, JpegXl, Zlib
register_codecs()
from scipy.spatial.transform import Rotation
from tactile_calibration.calibrate_tactile import compute_calibration_params, apply_calibration

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
            continue
        if isinstance(timestamps, dict) and 'timestamps' in timestamps:
            timestamps = timestamps['timestamps']
            
        if not os.path.exists(video_path):
            continue
            
        valid_len = len(timestamps)
        if mp4_ext == ".mp4":
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                continue
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            valid_len = min(valid_len, frame_count)
        elif mp4_ext == ".npz":
            try:
                with np.load(video_path) as data:
                    key = "depths" if "depths" in data else list(data.keys())[0]
                    frame_count = len(data[key])
            except Exception:
                continue
            valid_len = min(valid_len, frame_count)
            
        if valid_len <= 0:
            continue

        all_times.extend(timestamps[:valid_len])
        all_video_paths.extend([video_path] * valid_len)
        all_frame_indices.extend(range(valid_len))
    combined_values = list(zip(all_video_paths, all_frame_indices))
    return np.array(all_times), combined_values

def get_frame(video_path, frame_idx, video_caps):
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
        video_caps[video_path] = cap
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            raise ValueError(f"Failed to read frame {frame_idx} from {video_path} on retry.")
            
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame

def resize_with_black_padding(image, target_h=224, target_w=224, is_depth=False):
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

def record_loop(args):
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
        print(f"Error: No matching data folders found in {args.data_root} for cap_ids {args.cap_id}.")
        return

    handcap_file_list = ["handcap_" + f.split("_")[-1] for f in tracker_file_list]
    cap_ids_to_zip = [f.split("_")[-1] for f in tracker_file_list]
    
    out_replay_buffer = ReplayBuffer.create_empty_zarr(storage=zarr.MemoryStore())
    
    global_calib_params = None
    all_video_tasks = []
    buffer_start = 0
    saved_idx = args.start_episode_idx
    
    for batch_idx, (handcap_data_root, tracker_root, each_cap_id) in enumerate(zip(handcap_file_list, tracker_file_list, cap_ids_to_zip)):
        angle_calib_path = f"../Preprocess/angle_calib_params/gripper_calibration_interpolation50501.json"
        if not os.path.exists(angle_calib_path):
            print(f"Warning: Calib missing {angle_calib_path}. Using fallback identity.")
            as5600_preprocess = lambda x: x
            as5600_to_width = lambda x: x / 4096.0 * 0.08
        else:
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
                    def as5600_preprocess(x):
                        return x + 4096 if x < threshold else x
                values = list(map(as5600_preprocess, values))
                if values[0] > values[-1]:
                    values = [values[0]+200] + values + [values[-1]-200]
                else:
                    values = [values[0]-200] + values + [values[-1]+200]
                widths = list(widths)
                widths = [widths[0]] + widths + [widths[-1]]
                as5600_to_width = WrappedInterp1d(values, widths, extend_boundary=10)

        print(f"Processing data from {handcap_data_root} and {tracker_root}...")
        tracker_full_path = os.path.join(args.data_root, tracker_root)

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
                except:
                    continue
                ts_key = "timestamps"
                data_key = "data"
                htc_pose_times.extend(pose_data[ts_key])
                for p in pose_data[data_key]:
                    htc_pose_values.append(compute_pose_transform(p))
        htc_pose_times = np.array(htc_pose_times)
        
        phone_cam_folder = os.path.join(tracker_full_path, "phone_camera")
        phone_cam_times = np.array([])
        phone_cam_values = []
        if os.path.exists(phone_cam_folder):
            cam_jsons = [f for f in os.listdir(phone_cam_folder) if f.endswith('.json')]
            phone_cam_times, phone_cam_values = preload_video_data(phone_cam_folder, cam_jsons, "phone_camera_chunk_")
            
        phone_depth_folder = os.path.join(tracker_full_path, "phone_depth")
        phone_depth_times = np.array([])
        phone_depth_values = []
        if os.path.exists(phone_depth_folder):
            depth_jsons = [f for f in os.listdir(phone_depth_folder) if f.endswith('.json')]
            phone_depth_times, phone_depth_values = preload_video_data(phone_depth_folder, depth_jsons, "depth_chunk_", mp4_ext=".npz")
        
        handcap_full_path = os.path.join(args.data_root, handcap_data_root)
        try:
            episode_folders = [d for d in os.listdir(handcap_full_path) if os.path.isdir(os.path.join(handcap_full_path, d))]
        except Exception:
            episode_folders = []
            
        def extract_episode_num(name):
            nums = ''.join(filter(str.isdigit, name))
            return int(nums) if nums else 999999
            
        episode_folders.sort(key=lambda x: extract_episode_num(x))

        for ep_folder in episode_folders:
            ep_root = os.path.join(handcap_full_path, ep_folder)
            ep_num = extract_episode_num(ep_folder)
            
            angle_dataroot = os.path.join(ep_root, "angle")
            tactile_dataroot = os.path.join(ep_root, "tactile")
            wrist_dataroot = os.path.join(ep_root, "wrist")
            
            if not os.path.exists(wrist_dataroot):
                angle_dataroot = tactile_dataroot = wrist_dataroot = ep_root

            try:
                angle_file_list = [f for f in os.listdir(angle_dataroot) if f.startswith("angle_") and f.endswith(".json")]
                left_tactile_file_list = [f for f in os.listdir(tactile_dataroot) if f.startswith("lefttactile_") and f.endswith(".json")]
                right_tactile_file_list = [f for f in os.listdir(tactile_dataroot) if f.startswith("righttactile_") and f.endswith(".json")]
                wrist_file_list = [f for f in os.listdir(wrist_dataroot) if f.startswith("wrist") and f.endswith(".json")]
            except Exception as e:
                continue

            angle_times, angle_values = preload_data(angle_dataroot, angle_file_list, "angle_", "angles", "timestamps")
            left_tactile_times, left_tactile_values = preload_video_data(tactile_dataroot, left_tactile_file_list, "lefttactile_")
            right_tactile_times, right_tactile_values = preload_video_data(tactile_dataroot, right_tactile_file_list, "righttactile_")
            wrist_times, wrist_values = preload_video_data(wrist_dataroot, wrist_file_list, "wrist_")

            if len(wrist_times) == 0:
                continue

            if global_calib_params is None:
                calib_saved_path = os.path.join(args.data_root, "tactile_calib.json")
                if len(left_tactile_values) > 0 and len(right_tactile_values) > 0:
                    video_caps = {}
                    l_vid, l_idx = left_tactile_values[0]
                    left_calib_params = compute_calibration_params(cv2.cvtColor(get_frame(l_vid, l_idx, video_caps), cv2.COLOR_RGB2BGR))
                    r_vid, r_idx = right_tactile_values[0]
                    right_calib_params = compute_calibration_params(cv2.cvtColor(get_frame(r_vid, r_idx, video_caps), cv2.COLOR_RGB2BGR))
                    for cap in video_caps.values(): cap.release()
                    global_calib_params = (left_calib_params, right_calib_params)

            htc_pose_idx, angle_idx, ltactile_idx, rtactile_idx = 0, 0, 0, 0
            phone_cam_idx, phone_depth_idx = 0, 0
            
            first_wrist_time = wrist_times[0]
            last_wrist_time = wrist_times[-1]
            start_pose, _ = find_closest_timestamp(first_wrist_time, htc_pose_times, htc_pose_values, 0, threshold=100.0)
            end_pose, _ = find_closest_timestamp(last_wrist_time, htc_pose_times, htc_pose_values, htc_pose_idx, threshold=100.0)
            
            if start_pose is None or end_pose is None:
                continue

            episode_eef_pose = []
            episode_eef_rot_axis_angle = []
            episode_gripper_width = []
            
            ep_frames = []

            for time_idx, (wrist_video_path, wrist_frame_idx) in enumerate(wrist_values):
                each_time_step = wrist_times[time_idx]
                
                current_pose, htc_pose_idx = find_closest_timestamp(each_time_step, htc_pose_times, htc_pose_values, htc_pose_idx)
                agree_value, angle_idx = find_closest_timestamp(each_time_step, angle_times, angle_values, angle_idx)
                left_tactile_dict, ltactile_idx = find_closest_timestamp(each_time_step, left_tactile_times, left_tactile_values, ltactile_idx)
                right_tactile_dict, rtactile_idx = find_closest_timestamp(each_time_step, right_tactile_times, right_tactile_values, rtactile_idx)
                ph_cam_dict, phone_cam_idx = find_closest_timestamp(each_time_step, phone_cam_times, phone_cam_values, phone_cam_idx)
                ph_depth_dict, phone_depth_idx = find_closest_timestamp(each_time_step, phone_depth_times, phone_depth_values, phone_depth_idx)
                
                if current_pose is None or agree_value is None or left_tactile_dict is None or right_tactile_dict is None or ph_cam_dict is None or ph_depth_dict is None:
                    continue

                width_list = [[as5600_to_width(as5600_preprocess(x[0]))] for x in [agree_value]][0]

                episode_eef_pose.append(current_pose[:3])
                episode_eef_rot_axis_angle.append(current_pose[3:])
                episode_gripper_width.append(width_list[0])
                
                ep_frames.append({
                    'wrist_vid': wrist_video_path, 'wrist_idx': wrist_frame_idx,
                    'ltactile_vid': left_tactile_dict[0], 'ltactile_idx': left_tactile_dict[1],
                    'rtactile_vid': right_tactile_dict[0], 'rtactile_idx': right_tactile_dict[1],
                    'ph_cam_vid': ph_cam_dict[0], 'ph_cam_idx': ph_cam_dict[1],
                    'ph_depth_vid': ph_depth_dict[0], 'ph_depth_idx': ph_depth_dict[1]
                })

            if len(episode_eef_pose) > 0:
                robot_name = 'robot0'
                episode_data = dict()
                episode_data[robot_name + '_eef_pos'] = np.array(episode_eef_pose).astype(np.float32)
                episode_data[robot_name + '_eef_rot_axis_angle'] = np.array(episode_eef_rot_axis_angle).astype(np.float32)
                episode_data[robot_name + '_gripper_width'] = np.expand_dims(np.array(episode_gripper_width), axis=-1).astype(np.float32)
                episode_length = len(episode_eef_pose)
                episode_data[robot_name + '_demo_start_pose'] = np.repeat(start_pose[None, :], episode_length, axis=0).astype(np.float32)
                episode_data[robot_name + '_demo_end_pose'] = np.repeat(end_pose[None, :], episode_length, axis=0).astype(np.float32)

                out_replay_buffer.add_episode(data=episode_data, compressors=None)
                
                all_video_tasks.append({
                    'buffer_start': buffer_start,
                    'frames': ep_frames
                })
                buffer_start += episode_length
                print(f"Added episode {saved_idx} with {episode_length} frames.")
                saved_idx += 1

    print(f"Saving images to zarr...")
    img_compressor = JpegXl(level=99, numthreads=1)
    img_array = out_replay_buffer.data.require_dataset(
        name='camera0_rgb',
        shape=(out_replay_buffer['robot0_eef_pos'].shape[0],) + (224, 224) + (3,),
        chunks=(1,) + (224, 224) + (3,),
        compressor=img_compressor,
        dtype=np.uint8
    )
    
    tactile_array = out_replay_buffer.data.require_dataset(
        name="tactile_combined",
        shape=(out_replay_buffer['robot0_eef_pos'].shape[0],) + (224, 224) + (3,),
        chunks=(1,) + (224, 224) + (1,),
        compressor=img_compressor,
        dtype=np.uint8
    )
    
    phone_array = out_replay_buffer.data.require_dataset(
        name="camera0_phone_rgb",
        shape=(out_replay_buffer['robot0_eef_pos'].shape[0],) + (224, 224) + (3,),
        chunks=(1,) + (224, 224) + (3,),
        compressor=img_compressor,
        dtype=np.uint8
    )
    
    phone_depth_array = out_replay_buffer.data.require_dataset(
        name="camera0_phone_depth",
        shape=(out_replay_buffer['robot0_eef_pos'].shape[0],) + (224, 224) + (1,),
        chunks=(1,) + (224, 224) + (1,),
        compressor=None,
        dtype=np.float32
    )

    def process_episode(task_data, l_calib, r_calib):
        b_start = task_data['buffer_start']
        frames = task_data['frames']
        v_caps = {}
        npz_cache = {}
        
        def get_npz_frame(npz_path, frame_idx):
            if npz_path not in npz_cache:
                npz_cache.clear()
                data = np.load(npz_path)
                key = "depths" if "depths" in data else list(data.keys())[0]
                npz_cache[npz_path] = data[key]
            return npz_cache[npz_path][frame_idx]
        for i, f_dict in enumerate(frames):
            idx = b_start + i
            w_frame = get_frame(f_dict['wrist_vid'], f_dict['wrist_idx'], v_caps)
            l_frame = get_frame(f_dict['ltactile_vid'], f_dict['ltactile_idx'], v_caps)
            r_frame = get_frame(f_dict['rtactile_vid'], f_dict['rtactile_idx'], v_caps)
            
            p_frame = get_frame(f_dict['ph_cam_vid'], f_dict['ph_cam_idx'], v_caps)
            d_raw = get_npz_frame(f_dict['ph_depth_vid'], f_dict['ph_depth_idx'])
            
            if l_calib is not None: l_frame = apply_calibration(l_frame, l_calib)
            if r_calib is not None: r_frame = apply_calibration(r_frame, r_calib)
            
            img_array[idx] = resize_with_black_padding(w_frame, 224, 224)
            combined_tactile = np.concatenate([l_frame, r_frame], axis=1)
            tactile_array[idx] = resize_with_black_padding(combined_tactile, 224, 224)
            
            phone_array[idx] = resize_with_black_padding(p_frame, 224, 224)
            phone_depth_array[idx] = resize_with_black_padding(d_raw, 224, 224, is_depth=True).astype(np.float32)
            
        for cap in v_caps.values():
            cap.release()
        npz_cache.clear()
        return len(frames)

    l_calib = global_calib_params[0] if global_calib_params else None
    r_calib = global_calib_params[1] if global_calib_params else None

    num_workers = 2
    with tqdm(total=len(all_video_tasks), ncols=60) as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_episode, task, l_calib, r_calib) for task in all_video_tasks]
            for future in concurrent.futures.as_completed(futures):
                pbar.update(1)

    print("Dumped rgb camera and tactile")
    
    os.makedirs(args.output_root, exist_ok=True)
    zarr_path = os.path.join(args.output_root, "combined_data.zarr")
    print(f"Saving ReplayBuffer to {zarr_path}...")
    with zarr.ZipStore(zarr_path, mode='w') as zip_store:
        out_replay_buffer.save_to_store(store=zip_store)
    print("Done!")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--cap_id", type=str, default="all") 
    parser.add_argument("--data_root", type=str, default="handcap_data")
    parser.add_argument("--output_root", type=str, default="combined_data")
    parser.add_argument("--start_episode_idx", type=int, default=0)

    args = parser.parse_args()
    
    if os.path.exists(args.output_root):
        print(f"[WARNING] Output folder {args.output_root} already exists. Removing it...")
        shutil.rmtree(args.output_root)

    record_loop(args)
