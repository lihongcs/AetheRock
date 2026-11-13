# Portions adapted from the pyopenxr vive_tracker.py example.
# Copyright 2021 Christopher Bruns. Licensed under Apache-2.0.
# Modified by the AetheRock project for HandCap tracking.

import os
import xr
import cv2
import sys
import time
import ctypes
import msvcrt
import socket
import struct
import traceback
import datetime
import threading
import numpy as np
import pybullet as pb
from collections import deque
from config import *
from ctypes import cast, byref, POINTER
from argparse import ArgumentParser
import Utils.recordings_utils as utils
from scipy.spatial.transform import Rotation
from Postprocess.umi_utils.pose_util import *
import scipy.spatial.transform as st

from Sensor.htc_phone_process import HTCPhoneDataRecorder

collecting = False
running = True
episode_index = 1


HOST = '127.0.0.1'
PORT = 1337

# PeerTalkHeader: 4 x UInt32. Swift stores a,b,c as little-endian, body_size as big-endian.
# Total = 16 bytes
PEERTALK_HEADER_SIZE = 16

# Record3DHeader: 11 x UInt32 (little-endian, native ARM64)
# rgbWidth, rgbHeight, depthWidth, depthHeight, confWidth, confHeight,
# rgbSize, depthSize, confMapSize, miscSize, deviceType
RECORD3D_HEADER_SIZE = 44  # 11 * 4

# IntrinsicMatrixCoeffs: 4 x Float32 = 16 bytes
INTRINSIC_SIZE = 16

# CameraPose: 7 x Float32 (qx,qy,qz,qw, tx,ty,tz) = 28 bytes
POSE_SIZE = 28

# Phone data queue for async frame receiving
phone_frame_queue = deque(maxlen=2)  # Only keep latest frame
phone_queue_lock = threading.Lock()
phone_thread_running = True


import keyboard
def on_right_arrow(e):
    global episode_index
    episode_index = int(episode_index) + 1
    print(f"switch to {episode_index} episode...")
def on_left_arrow(e):
    global episode_index
    episode_index = max(0, int(episode_index) - 1)
    print(f"delete {episode_index} episode data...")
    os.system(f"rmdir /s /q data\\pose_{episode_index} 2>nul")


def compute_htc_relative_to_phone(
    phone_pose: np.ndarray,        # [x, y, z, rx, ry, rz]
    htc_pose: np.ndarray,          # [x, y, z, rx, ry, rz]
    calib_htc_pose: np.ndarray,    # [x, y, z, rx, ry, rz]  — htc 坐标系下的校准点
    calib_phone_pose: np.ndarray,  # [x, y, z, rx, ry, rz] — phone 坐标系下的校准点
) -> tuple[np.ndarray, np.ndarray]:
    """
    计算 htc 相对于 phone 的相对位姿（以 phone 坐标系为基准）。

    返回:
        relative_position: [x, y, z]
        relative_quat: [qx, qy, qz, qw]
    """
    # — 当前时刻 —
    t_phone = phone_pose[:3]
    R_phone = st.Rotation.from_rotvec(phone_pose[3:]).as_matrix()

    t_htc = htc_pose[:3]
    R_htc = st.Rotation.from_rotvec(htc_pose[3:]).as_matrix()

    # — 校准时刻（两个独立的坐标系）—
    # [DEBUG ac947b] Hypothesis A/B diagnostic inside function
    with open("debug-ac947b.log", "a") as lf:
        import json
        lf.write(json.dumps({
            "sessionId": "ac947b",
            "id": f"log_{time.time()}",
            "timestamp": int(time.time() * 1000),
            "location": "htc_tracker_phone_collect.py:88-92",
            "message": "calib_htc_pose and calib_phone_pose slices",
            "data": {
                "calib_htc_pose": calib_htc_pose.tolist() if calib_htc_pose is not None else None,
                "calib_htc_pose_shape": str(calib_htc_pose.shape) if calib_htc_pose is not None else None,
                "calib_htc_rot_slice": calib_htc_pose[3:].tolist() if calib_htc_pose is not None else None,
                "calib_phone_pose": calib_phone_pose.tolist() if calib_phone_pose is not None else None,
                "calib_phone_pose_shape": str(calib_phone_pose.shape) if calib_phone_pose is not None else None,
                "calib_phone_rot_slice": calib_phone_pose[3:].tolist() if calib_phone_pose is not None else None,
            },
            "runId": "debug-run",
            "hypothesisId": "A"
        }) + "\n")
    t_htc_c = calib_htc_pose[:3]
    R_htc_c = st.Rotation.from_rotvec(calib_htc_pose[3:]).as_matrix()

    t_phone_c = calib_phone_pose[:3]
    R_phone_c = st.Rotation.from_rotvec(calib_phone_pose[3:]).as_matrix()

    # — 相对旋转 —
    # ΔR_htc   = R_htc @ R_htc_c^T
    # ΔR_phone = R_phone @ R_phone_c^T
    # R_htc_phone = ΔR_htc @ ΔR_phone^T
    delta_R_htc   = R_htc @ R_htc_c.T
    delta_R_phone = R_phone @ R_phone_c.T
    R_htc_phone   = delta_R_htc @ delta_R_phone.T

    # — 相对位置 —
    t_htc_phone = (t_htc - t_htc_c) - R_htc_phone @ (t_phone - t_phone_c)

    # — 输出 —
    relative_quat = st.Rotation.from_matrix(R_htc_phone).as_quat()
    return t_htc_phone, relative_quat


def quat_to_rot(quat_pos):
    shape = quat_pos.shape[:-1]
    pose = np.zeros(shape+(6,), dtype=quat_pos.dtype)
    rot = st.Rotation.from_quat(quat_pos[...,3:])
    pos = quat_pos[...,:3]
    pose[...,:3] = pos
    pose[...,3:] = rot.as_rotvec()
    return pose

def compute_pose_transform(raw_pose):
    
    pose_mat = pose_to_mat(quat_to_rot(raw_pose))
    
    # x_rotation_clockwise_90 = Rotation.from_euler('x', -90, degrees=True).as_matrix()
    
    # transform_matrix = np.eye(4)
    # transform_matrix[:3, :3] = x_rotation_clockwise_90
    
    # pose_mat = transform_matrix @ pose_mat
    additional_transform_matrix = np.eye(4)
    additional_R_y = Rotation.from_euler('y', 90, degrees=True).as_matrix()
    additional_R_z = Rotation.from_euler('z', 180, degrees=True).as_matrix()
    transform_mat =  additional_R_y @ additional_R_z
    additional_transform_matrix[:3, :3] = transform_mat
    pose_mat_out = pose_mat @ additional_transform_matrix
    
    return mat_to_certain_pose_type(pose_mat_out, pose_type="rotvec")


def apply_phone_transform(pose_rot):
    """对 phone pose 绕 Y 轴逆时针旋转 90°"""
    pose_mat = pose_to_mat(quat_to_rot(pose_rot))
    additional_transform_matrix = np.eye(4)
    additional_transform_matrix[:3, :3] = Rotation.from_euler('y', -90, degrees=True).as_matrix()
    pose_mat = additional_transform_matrix @ pose_mat

    additional_transform_matrix = np.eye(4)
    additional_R_y = Rotation.from_euler('y', 90, degrees=True).as_matrix()
    additional_R_z = Rotation.from_euler('z', 180, degrees=True).as_matrix()
    transform_mat =  additional_R_y @ additional_R_z
    additional_transform_matrix[:3, :3] = transform_mat
    pose_mat_out = pose_mat @ additional_transform_matrix
    return mat_to_certain_pose_type(pose_mat_out, pose_type="rotvec")

def recv_exact(sock, n):
    data = b''
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise ConnectionResetError("Connection closed by server")
        data += chunk
    return data


def phone_receiver_thread(sock):
    sock.settimeout(0.1)  # 短超时，允许定期检查线程是否应该停止
    
    while phone_thread_running:
        try:
            # 放入队列
            # --- 1. PeerTalk header (16 bytes) ---
            pt_raw = recv_exact(sock, PEERTALK_HEADER_SIZE)
            # a, b, c are little-endian UInt32; body_size is big-endian UInt32
            pt_a, pt_b, pt_c = struct.unpack_from('<III', pt_raw, 0)
            body_size = struct.unpack_from('>I', pt_raw, 12)[0]

            if body_size == 0:
                continue

            # --- 2. Full body ---
            body = recv_exact(sock, body_size)
            offset = 0

            # --- 3. Record3D header (44 bytes, little-endian) ---
            if len(body) < offset + RECORD3D_HEADER_SIZE:
                print(f"Body too short for Record3D header: {len(body)}")
                continue
            (rgb_w, rgb_h, depth_w, depth_h, conf_w, conf_h,
                rgb_size, depth_size, conf_size, misc_size, device_type
                ) = struct.unpack_from('<11I', body, offset)
            offset += RECORD3D_HEADER_SIZE

            # --- 4. Intrinsic matrix (16 bytes) ---
            if len(body) < offset + INTRINSIC_SIZE:
                continue
            fx, fy, tx, ty = struct.unpack_from('<4f', body, offset)
            offset += INTRINSIC_SIZE

            # --- 5. Camera pose (28 bytes) ---
            if len(body) < offset + POSE_SIZE:
                continue
            qx, qy, qz, qw, tx_p, ty_p, tz_p = struct.unpack_from('<7f', body, offset)
            offset += POSE_SIZE

            # --- 6. RGB JPEG data ---
            if rgb_size == 0:
                print(f"Warning: rgb_size is 0! body={len(body)}, offset={offset}")
                continue
            if len(body) < offset + rgb_size:
                print(f"Body too short for RGB: need {offset + rgb_size}, have {len(body)}")
                continue

            rgb_jpeg = body[offset:offset + rgb_size]
            offset += rgb_size

            # Decode JPEG
            nparr = np.frombuffer(rgb_jpeg, np.uint8)
            rgb_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            frame_data = {
                'image': rgb_img,
                'position': [tx_p, ty_p, tz_p],
                'quaternion': [qw, qx, qy, qz],
                'timestamp': time.time()
            }
            
            with phone_queue_lock:
                phone_frame_queue.append(frame_data)
                
        except socket.timeout:
            continue
        except Exception as e:
            if phone_thread_running:
                print(f"Phone receiver error: {e}")
            continue


def try_get_phone_frame():
    """尝试从队列获取最新的Phone数据（非阻塞）"""
    with phone_queue_lock:
        if phone_frame_queue:
            return phone_frame_queue[-1]  # 获取最新帧
        return None

def quaternion_to_rotation_matrix(q):
    rot = st.Rotation.from_quat(q)
    return rot.as_matrix()

class DataVisualizer:
    """可视化器：显示 3D 透视轨迹 + 末端姿态轴（上=HTC，下=Phone）"""
    def __init__(self, width=480, height=960):
        # 1. 尺寸配置
        self.w = width
        self.h = height
        self.h_top = int(height * 0.5)
        self.h_bot = height - self.h_top
        
        # 每部分再分 4 列：三视图(各1/3宽) + 3D透视(1/3宽)
        self.sub_w = self.w // 4
        self.h_pos = self.h_top
        self.h_rot = self.h_top // 2

        # 状态记录
        self.was_recording = False
        self.htc_pose_history = []
        self.phone_pose_history = []

        # 缓存的静态图片 (用于停止录制后固定显示)
        self.frozen_frame = None

        # 最小量程配置
        self.MIN_RANGE_POS = 0.5
        self.MIN_RANGE_ROT = 0.5

        # 3D 透视投影参数（俯视 30°）
        self.PITCH = np.radians(30)
        self.AXIS_LEN_FACTOR = 0.12   # 轴长占格子尺寸的比例

    def update(self, htc_pose, phone_pose, is_recording, is_calibrating):

        # A. [开始录制] 时刻 (Rising Edge) - 只要 is_recording 变 True 就清空
        if is_recording and not self.was_recording:
            self.htc_pose_history = []
            self.phone_pose_history = []
            self.frozen_frame = None

        # B. [停止录制] 时刻 (Falling Edge) - 非校准模式下才冻结
        elif not is_recording and self.was_recording and not is_calibrating:
            self.frozen_frame = self._render_canvas()

        # C. [校准中] - 追加 history 用于实时显示，不冻结
        if is_calibrating:
            self.htc_pose_history.append(htc_pose)
            self.phone_pose_history.append(phone_pose)

        # D. [录制中] - 追加 history 用于录制
        if is_recording and not is_calibrating:
            self.htc_pose_history.append(htc_pose)
            self.phone_pose_history.append(phone_pose)

        self.was_recording = is_recording

    def draw(self):
        """
        主绘图接口：
        - 如果有缓存的冻结帧（即录制结束状态），直接返回图片，极快。
        - 如果正在录制，实时渲染。
        - 如果是初始状态，渲染空背景。
        """
        if self.frozen_frame is not None:
            return self.frozen_frame

        # 如果没有冻结帧（说明正在录制，或者还没开始过），则实时渲染
        return self._render_canvas()

    def _render_canvas(self):
        """内部渲染逻辑：只在录制时或生成冻结帧时调用"""
        canvas = np.zeros((self.h, self.w, 3), dtype=np.uint8)

        if not self.htc_pose_history and not self.phone_pose_history:
            cv2.putText(canvas, "Ready to Record", (self.w//2 - 60, self.h//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100,100,100), 1)
            return canvas

        sub_w = self.sub_w
        h_pos = self.h_pos

        # ==================== 上半部分：HTC ====================
        htc_valid_poses = [p for p in self.htc_pose_history if p is not None]
        if len(htc_valid_poses) > 1:
            htc_xs = [p[0] for p in htc_valid_poses]
            htc_ys = [p[1] for p in htc_valid_poses]
            htc_zs = [p[2] for p in htc_valid_poses]
            htc_min_x, htc_max_x, htc_rng_x = self._get_range_lims(htc_xs, self.MIN_RANGE_POS)
            htc_min_y, htc_max_y, htc_rng_y = self._get_range_lims(htc_ys, self.MIN_RANGE_POS)
            htc_min_z, htc_max_z, htc_rng_z = self._get_range_lims(htc_zs, self.MIN_RANGE_POS)

            # 绘制辅助函数
            def plot_trace(sx, sy, data_h, data_v, min_h, rng_h, min_v, rng_v, label_h, label_v, color):
                self._draw_grid(canvas, sx, sy, sub_w, h_pos, min_v, min_v + rng_v, f"{label_h}-{label_v}")
                pts = []
                for i in range(len(data_h)):
                    px = int(sx + 10 + ((data_h[i] - min_h) / rng_h) * (sub_w - 20))
                    py = int(sy + h_pos - 10 - ((data_v[i] - min_v) / rng_v) * (h_pos - 20))
                    pts.append([px, py])
                cv2.polylines(canvas, [np.array(pts)], False, color, 1)
                cv2.circle(canvas, tuple(pts[0]), 3, (0, 255, 0), -1)
                cv2.circle(canvas, tuple(pts[-1]), 3, (0, 0, 255), -1)

            # HTC 三视图
            plot_trace(0, 0, htc_xs, htc_zs, htc_min_x, htc_rng_x, htc_min_z, htc_rng_z, "X", "Z", (0, 255, 255))
            plot_trace(sub_w, 0, htc_xs, htc_ys, htc_min_x, htc_rng_x, htc_min_y, htc_rng_y, "X", "Y", (0, 255, 255))
            plot_trace(2*sub_w, 0, htc_zs, htc_ys, htc_min_z, htc_rng_z, htc_min_y, htc_rng_y, "Z", "Y", (0, 255, 255))
            
            # HTC 3D 透视 + 姿态轴
            self._draw_3d_trajectory(canvas, 3*sub_w, 0, sub_w, h_pos,
                                    htc_valid_poses, (0, 255, 255), "3D")
            cv2.putText(canvas, "HTC Tracker", (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        # ==================== 下半部分：Phone ====================
        phone_valid_poses = [p for p in self.phone_pose_history if p is not None]
        if len(phone_valid_poses) > 1:
            phone_xs = [p[0] for p in phone_valid_poses]
            phone_ys = [p[1] for p in phone_valid_poses]
            phone_zs = [p[2] for p in phone_valid_poses]
            phone_min_x, phone_max_x, phone_rng_x = self._get_range_lims(phone_xs, self.MIN_RANGE_POS)
            phone_min_y, phone_max_y, phone_rng_y = self._get_range_lims(phone_ys, self.MIN_RANGE_POS)
            phone_min_z, phone_max_z, phone_rng_z = self._get_range_lims(phone_zs, self.MIN_RANGE_POS)

            sy_phone = self.h_top

            def plot_trace(sx, sy, data_h, data_v, min_h, rng_h, min_v, rng_v, label_h, label_v, color):
                self._draw_grid(canvas, sx, sy, sub_w, h_pos, min_v, min_v + rng_v, f"{label_h}-{label_v}")
                pts = []
                for i in range(len(data_h)):
                    px = int(sx + 10 + ((data_h[i] - min_h) / rng_h) * (sub_w - 20))
                    py = int(sy + h_pos - 10 - ((data_v[i] - min_v) / rng_v) * (h_pos - 20))
                    pts.append([px, py])
                cv2.polylines(canvas, [np.array(pts)], False, color, 1)
                cv2.circle(canvas, tuple(pts[0]), 3, (0, 255, 0), -1)
                cv2.circle(canvas, tuple(pts[-1]), 3, (0, 0, 255), -1)

            # Phone 三视图
            plot_trace(0, sy_phone, phone_xs, phone_zs, phone_min_x, phone_rng_x, phone_min_z, phone_rng_z, "X", "Z", (255, 200, 100))
            plot_trace(sub_w, sy_phone, phone_xs, phone_ys, phone_min_x, phone_rng_x, phone_min_y, phone_rng_y, "X", "Y", (255, 200, 100))
            plot_trace(2*sub_w, sy_phone, phone_zs, phone_ys, phone_min_z, phone_rng_z, phone_min_y, phone_rng_y, "Z", "Y", (255, 200, 100))
            
            # Phone 3D 透视 + 姿态轴
            self._draw_3d_trajectory(canvas, 3*sub_w, sy_phone, sub_w, h_pos,
                                    phone_valid_poses, (255, 200, 100), "3D")
            cv2.putText(canvas, "Phone Pose", (10, sy_phone + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 100), 1)

        # 中间分隔线
        cv2.line(canvas, (0, self.h_top), (self.w, self.h_top), (80, 80, 80), 1)
        # 三视图与3D之间
        cv2.line(canvas, (3*sub_w, 0), (3*sub_w, self.h_top), (80, 80, 80), 1)
        cv2.line(canvas, (3*sub_w, self.h_top), (3*sub_w, self.h), (80, 80, 80), 1)
        return canvas

    def _get_range_lims(self, data_arr, min_range_val):
        if not data_arr: return 0, 1, 1
        filtered = [v for v in data_arr if v is not None]
        if not filtered: return 0, 1, 1
        v_min, v_max = min(filtered), max(filtered)
        current_range = v_max - v_min
        if current_range < min_range_val:
            center = (v_max + v_min) / 2
            v_max = center + min_range_val / 2
            v_min = center - min_range_val / 2
            current_range = min_range_val
        return v_min, v_max, current_range

    def _draw_grid(self, img, x, y, w, h, v_min, v_max, title):
        cv2.rectangle(img, (x, y), (x+w, y+h), (30, 30, 30), -1)
        cv2.rectangle(img, (x, y), (x+w, y+h), (80, 80, 80), 1)
        cv2.putText(img, title, (x+5, y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    def _draw_3d_trajectory(self, canvas, sx, sy, sw, sh,
                           valid_poses, color, label):
        """绘制 3D 透视轨迹 + 末端姿态轴"""
        self._draw_grid(canvas, sx, sy, sw, sh, 0, 1, label)
        if len(valid_poses) < 2:
            return

        # 提取位置和量程
        xs = [p[0] for p in valid_poses]
        ys = [p[1] for p in valid_poses]
        zs = [p[2] for p in valid_poses]
        min_x, max_x, rng_x = self._get_range_lims(xs, self.MIN_RANGE_POS)
        min_y, max_y, rng_y = self._get_range_lims(ys, self.MIN_RANGE_POS)
        min_z, max_z, rng_z = self._get_range_lims(zs, self.MIN_RANGE_POS)

        # 透视投影：绕 X 轴旋转 PITCH 度
        cos_p = np.cos(self.PITCH)
        sin_p = np.sin(self.PITCH)
        rng_y_proj = (max_y * cos_p - min_z * sin_p) - (min_y * cos_p - max_z * sin_p)
        rng_y_proj = max(abs(rng_y_proj), 1e-9)
        rng_z_proj = (max_y * sin_p + max_z * cos_p) - (min_y * sin_p + min_z * cos_p)
        rng_z_proj = max(abs(rng_z_proj), 1e-9)

        def project(x, y, z):
            y_proj = cos_p * y - sin_p * z
            px = int(sx + 15 + ((x - min_x) / rng_x) * (sw - 30))
            py = int(sy + sh - 15 - ((y_proj - (min_y * cos_p - max_z * sin_p)) / rng_y_proj) * (sh - 30))
            return px, py

        # 绘制轨迹线
        pts = [project(p[0], p[1], p[2]) for p in valid_poses]
        cv2.polylines(canvas, [np.array(pts)], False, color, 2)
        # 起点（绿）和终点（红）
        cv2.circle(canvas, pts[0], 4, (0, 255, 0), -1)
        cv2.circle(canvas, pts[-1], 4, (0, 0, 255), -1)

        # 末端姿态轴：提取旋转矩阵
        last = valid_poses[-1]
        if len(last) >= 6 and last[3:6] is not None:
            rv = last[3:6]
            rot = st.Rotation.from_rotvec(rv)
            R = rot.as_matrix()
            end_px, end_py = pts[-1]
            axis_len = int(sh * self.AXIS_LEN_FACTOR)

            # 绘制 X 轴（红）Y 轴（绿）Z 轴（蓝）
            for col, bgr, vec_name in [(R[:, 0], (0, 0, 255), 'X'),
                                        (R[:, 1], (0, 255, 0), 'Y'),
                                        (R[:, 2], (255, 0, 0), 'Z')]:
                dx, dy, dz = col
                ax = int(dx * axis_len * rng_x / (rng_x + rng_y + rng_z + 1e-9) * 3)
                ay = int((cos_p * dy - sin_p * dz) * axis_len * rng_y_proj / (rng_y_proj + rng_z_proj + 1e-9) * 3)
                cv2.arrowedLine(canvas, (end_px, end_py),
                                (end_px + ax, end_py - ay),
                                bgr, 2, tipLength=0.3)

def keyboard_listener():
    global collecting, running, episode_index
    
    while running:
        if msvcrt.kbhit():
            key = msvcrt.getch()
            if key == b' ':
                collecting = not collecting
                if collecting:
                    print("Beginning...")
                else:
                    print("Paused...")
            elif key in [b'q', b'Q']:
                collecting = False
                print("collected finished.")
            elif key == b'\x1b':  # ESC
                collecting = False
                running = False
                data_recorder.stop()
                print("ESC pressed. Exiting.")
            elif key == b'\xe0':
                key2 = msvcrt.getch()
                if key2 == b'K':
                    print(f"delete {episode_index} episode data...")
                    os.system(f"rmdir /s /q data\\pose_{episode_index} 2>nul")
                elif key2 == b'M': 
                    # 修复：确保 episode_index 为整数类型进行运算
                    episode_index = int(episode_index) + 1
                    print(f"switch to {episode_index} episode...")
        time.sleep(0.01)


class ContextObject(object):
    def __init__(
            self,
            instance_create_info: xr.InstanceCreateInfo = xr.InstanceCreateInfo(),
            session_create_info: xr.SessionCreateInfo = xr.SessionCreateInfo(),
            reference_space_create_info: xr.ReferenceSpaceCreateInfo = xr.ReferenceSpaceCreateInfo(),
            view_configuration_type: xr.ViewConfigurationType = xr.ViewConfigurationType.PRIMARY_STEREO,
            environment_blend_mode=xr.EnvironmentBlendMode.OPAQUE,
            form_factor=xr.FormFactor.HEAD_MOUNTED_DISPLAY,
    ):
        self._instance_create_info = instance_create_info
        self.instance = None
        self._session_create_info = session_create_info
        self.session = None
        self.session_state = xr.SessionState.IDLE
        self._reference_space_create_info = reference_space_create_info
        self.view_configuration_type = view_configuration_type
        self.environment_blend_mode = environment_blend_mode
        self.form_factor = form_factor
        self.graphics = None
        self.graphics_binding_pointer = None
        self.action_sets = []
        self.render_layers = []
        self.swapchains = []
        self.swapchain_image_ptr_buffers = []
        self.swapchain_image_buffers = []  # Keep alive
        self.exit_render_loop = False
        self.request_restart = False  # TODO: do like hello_xr
        self.session_is_running = False

    def __enter__(self):
        self.instance = xr.create_instance(
            create_info=self._instance_create_info,
        )
        self.system_id = xr.get_system(
            instance=self.instance,
            get_info=xr.SystemGetInfo(
                form_factor=self.form_factor,
            ),
        )

        if self._session_create_info.next is not None:
            self.graphics_binding_pointer = self._session_create_info.next

        self._session_create_info.system_id = self.system_id
        self.session = xr.create_session(
            instance=self.instance,
            create_info=self._session_create_info,
        )
        self.space = xr.create_reference_space(
            session=self.session,
            create_info=self._reference_space_create_info
        )
        self.default_action_set = xr.create_action_set(
            instance=self.instance,
            create_info=xr.ActionSetCreateInfo(
                action_set_name="default_action_set",
                localized_action_set_name="Default Action Set",
                priority=0,
            ),
        )
        self.action_sets.append(self.default_action_set)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.default_action_set is not None:
            xr.destroy_action_set(self.default_action_set)
            self.default_action_set = None
        if self.space is not None:
            xr.destroy_space(self.space)
            self.space = None
        if self.session is not None:
            xr.destroy_session(self.session)
            self.session = None
        if self.graphics is not None:
            self.graphics.destroy()
            self.graphics = None
        if self.instance is not None:
            xr.destroy_instance(self.instance)
            self.instance = None

    def frame_loop(self):
        xr.attach_session_action_sets(
            session=self.session,
            attach_info=xr.SessionActionSetsAttachInfo(
                count_action_sets=len(self.action_sets),
                action_sets=(xr.ActionSet * len(self.action_sets))(
                    *self.action_sets
                )
            ),
        )
        while True:
            self.exit_render_loop = False
            self.poll_xr_events()
            if self.exit_render_loop:
                break
            if self.session_is_running:
                if self.session_state in (
                        xr.SessionState.READY,
                        xr.SessionState.SYNCHRONIZED,
                        xr.SessionState.VISIBLE,
                        xr.SessionState.FOCUSED,
                ):
                    frame_state = xr.wait_frame(self.session)
                    xr.begin_frame(self.session)
                    self.render_layers = []

                    yield frame_state

                    xr.end_frame(
                        self.session,
                        frame_end_info=xr.FrameEndInfo(
                            display_time=frame_state.predicted_display_time,
                            environment_blend_mode=self.environment_blend_mode,
                            layers=self.render_layers,
                        )
                    )
            else:
                # Throttle loop since xrWaitFrame won't be called.
                time.sleep(0.250)

    def poll_xr_events(self):
        self.exit_render_loop = False
        self.request_restart = False
        while True:
            try:
                event_buffer = xr.poll_event(self.instance)
                try:
                    event_type = xr.StructureType(event_buffer.type)
                except ValueError as e:
                    print(f"Unknown event type: {event_buffer.type}, skipping...")
                    continue
                # event_type = xr.StructureType(event_buffer.type)
                if event_type == xr.StructureType.EVENT_DATA_INSTANCE_LOSS_PENDING:
                    # still handle rest of the events instead of immediately quitting
                    self.exit_render_loop = True
                    self.request_restart = True
                elif event_type == xr.StructureType.EVENT_DATA_SESSION_STATE_CHANGED \
                        and self.session is not None:
                    event = cast(
                        byref(event_buffer),
                        POINTER(xr.EventDataSessionStateChanged)).contents
                    self.session_state = xr.SessionState(event.state)
                    if self.session_state == xr.SessionState.READY:
                        xr.begin_session(
                            session=self.session,
                            begin_info=xr.SessionBeginInfo(
                                self.view_configuration_type,
                            ),
                        )
                        self.session_is_running = True
                    elif self.session_state == xr.SessionState.STOPPING:
                        self.session_is_running = False
                        xr.end_session(self.session)
                    elif self.session_state == xr.SessionState.EXITING:
                        self.exit_render_loop = True
                        self.request_restart = False
                    elif self.session_state == xr.SessionState.LOSS_PENDING:
                        self.exit_render_loop = True
                        self.request_restart = True
                elif event_type == xr.StructureType.EVENT_DATA_VIVE_TRACKER_CONNECTED_HTCX:
                    vive_tracker_connected = cast(byref(event_buffer), POINTER(xr.EventDataViveTrackerConnectedHTCX)).contents
                    paths = vive_tracker_connected.paths.contents
                    persistent_path_str = xr.path_to_string(self.instance, paths.persistent_path)
                    if paths.role_path != xr.NULL_PATH:
                        role_path_str = xr.path_to_string(self.instance, paths.role_path)
                    else:
                        pass
                elif event_type == xr.StructureType.EVENT_DATA_INTERACTION_PROFILE_CHANGED:
                    # print("data interaction profile changed")
                    # TODO:
                    pass
            except xr.EventUnavailable:
                break

    def view_loop(self, frame_state):
        if frame_state.should_render:
            layer = xr.CompositionLayerProjection(space=self.space)
            view_state, views = xr.locate_views(
                session=self.session,
                view_locate_info=xr.ViewLocateInfo(
                    view_configuration_type=self.view_configuration_type,
                    display_time=frame_state.predicted_display_time,
                    space=self.space,
                )
            )
            num_views = len(views)
            projection_layer_views = tuple(xr.CompositionLayerProjectionView() for _ in range(num_views))

            vsf = view_state.view_state_flags
            if (vsf & xr.VIEW_STATE_POSITION_VALID_BIT == 0
                    or vsf & xr.VIEW_STATE_ORIENTATION_VALID_BIT == 0):
                return  # There are no valid tracking poses for the views.
            for view_index, view in enumerate(views):
                view_swapchain = self.swapchains[view_index]
                swapchain_image_index = xr.acquire_swapchain_image(
                    swapchain=view_swapchain.handle,
                    acquire_info=xr.SwapchainImageAcquireInfo(),
                )
                xr.wait_swapchain_image(
                    swapchain=view_swapchain.handle,
                    wait_info=xr.SwapchainImageWaitInfo(timeout=xr.INFINITE_DURATION),
                )
                layer_view = projection_layer_views[view_index]
                assert layer_view.type == xr.StructureType.COMPOSITION_LAYER_PROJECTION_VIEW
                layer_view.pose = view.pose
                layer_view.fov = view.fov
                layer_view.sub_image.swapchain = view_swapchain.handle
                layer_view.sub_image.image_rect.offset[:] = [0, 0]
                layer_view.sub_image.image_rect.extent[:] = [
                    view_swapchain.width, view_swapchain.height, ]
                swapchain_image_ptr = self.swapchain_image_ptr_buffers[view_index][swapchain_image_index]
                swapchain_image = cast(swapchain_image_ptr, POINTER(xr.SwapchainImageOpenGLKHR)).contents
                assert layer_view.sub_image.image_array_index == 0  # texture arrays not supported.
                color_texture = swapchain_image.image
                self.graphics.begin_frame(layer_view, color_texture)

                yield view

                self.graphics.end_frame()
                xr.release_swapchain_image(
                    swapchain=view_swapchain.handle,
                    release_info=xr.SwapchainImageReleaseInfo()
                )
            layer.views = projection_layer_views
            self.render_layers.append(byref(layer))



def main(args):
    frequency = args.fps
    global episode_index, collecting, running
    wf_receive_ts = time.time()
    
    formatted_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    data_recorder = HTCPhoneDataRecorder(save_dir = f"data/{args.cap_id}_{formatted_time}",
                                        fps = args.fps,
                                        chunksize = 150,
                                        camera_resolution = (960, 720))
    data_recorder.start()
    viz_width = 480
    viz_height = 960
    viz_scale = 100.0
    camera_distance = 2.0
    camera_elevation = 25
    camera_azimuth = 135
    max_points = 500

    # 可视化数据
    tracker_world_positions = []
    relative_positions = []

    position = None
    orientation = None
    tracker_position = None
    tracker_orientation = None

    # 创建DataVisualizer用于位置和旋转可视化（三视图+时序图布局）
    # 位置可视化器
    pos_viz_width = 720
    pos_viz_height = 960
    visualizer = DataVisualizer(width=pos_viz_width, height=pos_viz_height)

    # flag to indicate if need to calibrate world coordinate system
    calibrating = False

    first = True
    first_2 = True
    first_3 = True
    episode_index = input("Enter episode index: ")
    print("******** Control Instructions *********************")
    print("******** Press 'space' to start collecting ********")
    print("******** Press 'c' to calibrate world coordinate system ************************")
    print("******** Press 'q' to quit ************************")
    print("******** Press '<-' delete collected data *********")
    print("******** Press '->' to next episode ***************")
    keyboard_thread = threading.Thread(target=keyboard_listener, daemon=True)
    keyboard_thread.start()
    
    print(f"Connecting to AnySense at {HOST}:{PORT}...")
    print("Make sure 'iproxy 1337 1337' is running in another terminal!")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        print("✅ Connected! Checking phone status...")
        
        # Set socket timeout for receiving
        s.settimeout(30.0)
        
        # Wait for phone to be in USB mode and recording
        phone_ready = False
        while not phone_ready:
            try:
                print("Waiting for phone data (USB mode + recording)...")
                pt_raw = recv_exact(s, PEERTALK_HEADER_SIZE)
                pt_a, pt_b, pt_c = struct.unpack_from('<III', pt_raw, 0)
                body_size = struct.unpack_from('>I', pt_raw, 12)[0]
                
                if body_size == 0:
                    print("  -> Phone connected but NOT recording (body_size=0)")
                    continue
                
                body = recv_exact(s, body_size)
                if len(body) < RECORD3D_HEADER_SIZE:
                    print(f"  -> Invalid data received")
                    continue
                
                (rgb_w, rgb_h, depth_w, depth_h, conf_w, conf_h,
                rgb_size, depth_size, conf_size, misc_size, device_type
                ) = struct.unpack_from('<11I', body, 0)
                
                if depth_size > 0:
                    print("✅ Depth data received")
                else:
                    print("❌ No depth data received")

                # device_type: 0 = LiDAR iPhone, 1 = LiDAR iPad, 2 = Depth camera iPhone, etc.
                # When in USB mode and recording, rgb_size should be > 0
                if rgb_size > 0:
                    # USB mode typically has device_type 0-2
                    if device_type in [0, 1, 2]:
                        print(f"✅ Phone ready! USB mode detected (device_type={device_type}), recording=True")
                        phone_ready = True
                    else:
                        print(f"  -> Phone connected but NOT in USB mode (device_type={device_type})")
                else:
                    print("  -> Phone connected but NOT recording (rgb_size=0)")
                    
            except socket.timeout:
                print("⏱️ Timeout waiting for phone data. Make sure:")
                print("   1. iPhone is connected via USB")
                print("   2. iPhone is in USB mode (not WiFi)")
                print("   3. You have pressed the Record button")
                print("   Retrying...")
                continue
            except Exception as e:
                print(f"Error checking phone status: {e}")
                time.sleep(1)
                continue
        
        print("✅ Phone verified! Starting main loop...")
        
        global phone_thread_running
        phone_thread_running = True
        receiver_thread = threading.Thread(target=phone_receiver_thread, args=(s,), daemon=True)
        receiver_thread.start()
        print("Phone receiver thread started!")

        print("Creating OpenXR context...")
        print("  -> Creating instance...")
        try:
            with ContextObject(
                instance_create_info=xr.InstanceCreateInfo(
                    enabled_extension_names=[
                        xr.MND_HEADLESS_EXTENSION_NAME,
                        "XR_HTCX_vive_tracker_interaction",
                    ],
                ),
            ) as context:
                print("  -> Instance created, waiting for session...")
                instance = context.instance
                session = context.session
                print("OpenXR context created successfully!")

                enumerateViveTrackerPathsHTCX = cast(
                    xr.get_instance_proc_addr(instance, "xrEnumerateViveTrackerPathsHTCX"),
                    xr.PFN_xrEnumerateViveTrackerPathsHTCX
                )

                role_strings = [
                    "handheld_object", "left_foot", "right_foot", "left_shoulder", "right_shoulder",
                    "left_elbow", "right_elbow", "left_knee", "right_knee", "waist", "chest",
                    "camera", "keyboard",
                ]
                role_path_strings = [f"/user/vive_tracker_htcx/role/{role}" for role in role_strings]
                role_paths = (xr.Path * len(role_path_strings))(*[xr.string_to_path(instance, role_string) for role_string in role_path_strings])
                pose_action = xr.create_action(
                    action_set=context.default_action_set,
                    create_info=xr.ActionCreateInfo(
                        action_type=xr.ActionType.POSE_INPUT,
                        action_name="tracker_pose",
                        localized_action_name="Tracker Pose",
                        count_subaction_paths=len(role_paths),
                        subaction_paths=role_paths,
                    ),
                )
                suggested_binding_paths = (xr.ActionSuggestedBinding * len(role_path_strings))(
                    *[xr.ActionSuggestedBinding(pose_action, xr.string_to_path(instance, f"{role_path_string}/input/grip/pose")) for role_path_string in role_path_strings]
                )
                xr.suggest_interaction_profile_bindings(instance=instance, suggested_bindings=xr.InteractionProfileSuggestedBinding(
                    interaction_profile=xr.string_to_path(instance, "/interaction_profiles/htc/vive_tracker_htcx"),
                    count_suggested_bindings=len(suggested_binding_paths), suggested_bindings=suggested_binding_paths,
                ))
                tracker_action_spaces = (xr.Space * len(role_paths))(
                    *[xr.create_action_space(session=session, create_info=xr.ActionSpaceCreateInfo(action=pose_action, subaction_path=role_path)) for role_path in role_paths]
                )

                n_paths = ctypes.c_uint32(0)
                print("Calling enumerateViveTrackerPathsHTCX to get count...")
                result = enumerateViveTrackerPathsHTCX(instance, 0, byref(n_paths), None)
                print(f"enumerateViveTrackerPathsHTCX returned: {xr.Result(result)}, count: {n_paths.value}")
                if xr.check_result(result).is_exception():
                    raise result
                vive_tracker_paths = (xr.ViveTrackerPathsHTCX * n_paths.value)(*([xr.ViveTrackerPathsHTCX()] * n_paths.value))
                print("Calling enumerateViveTrackerPathsHTCX to get paths...")
                result = enumerateViveTrackerPathsHTCX(instance, n_paths, byref(n_paths), vive_tracker_paths)
                print(f"enumerateViveTrackerPathsHTCX paths returned: {xr.Result(result)}")

                print("Starting main loop - entering frame_loop()...")
                print("NOTE: If session doesn't enter FOCUSED state, you may need to:")
                print("   1. Start a VR app that uses OpenXR")
                print("   2. Or check if Vive Tracker is properly connected")
                print("   3. Ensure SteamVR or other OpenXR runtime is running")

                # Debug: Wait briefly and check session state before entering loop
                print(f"Pre-check: session_state={context.session_state}, session_is_running={context.session_is_running}")

                # Initialize a flag to track first iteration
                first_loop_iteration = True
                current_ts = time.time()
                last_print = time.time()
                frame_count = 0
                raw_xyz = None
                raw_rotation = None
                
                try:
                    from ctypes import windll
                    hwnd = windll.user32.FindWindowW(None, "Combined View")
                    if hwnd:
                        windll.user32.SetWindowPos(hwnd, -1, 0, 0, 0, 0, 0x0001 | 0x0002)
                except Exception:
                    pass

                for frame_state in context.frame_loop():

                    if first_loop_iteration:
                        print(f"First frame loop iteration - session_state={context.session_state}")
                        first_loop_iteration = False

                    if not running:
                        break

                    # Debug: print session state if not focused
                    if context.session_state != xr.SessionState.FOCUSED:
                        print(f"Waiting for session to be FOCUSED... current state: {context.session_state}, session_is_running={context.session_is_running}")
                        time.sleep(0.5)  # Avoid flooding output
                        continue

                    # Debug: notify when session becomes focused for first time
                    if not getattr(context, '_first_focus_logged', False):
                        print(f"✅ Session entered FOCUSED state!")
                        context._first_focus_logged = True

                    # 只在第一帧打印，避免刷屏
                    if first_loop_iteration:
                        print(f"Session state: {context.session_state}, processing frame...")

                    if context.session_state == xr.SessionState.FOCUSED:
                        active_action_set = xr.ActiveActionSet(action_set=context.default_action_set, subaction_path=xr.NULL_PATH)
                        xr.sync_actions(session=session, sync_info=xr.ActionsSyncInfo(count_active_action_sets=1, active_action_sets=ctypes.pointer(active_action_set)))

                        n_paths = ctypes.c_uint32(0)
                        result = enumerateViveTrackerPathsHTCX(instance, 0, byref(n_paths), None)
                        if xr.check_result(result).is_exception():
                            raise result
                        vive_tracker_paths = (xr.ViveTrackerPathsHTCX * n_paths.value)(*([xr.ViveTrackerPathsHTCX()] * n_paths.value))
                        result = enumerateViveTrackerPathsHTCX(instance, n_paths, byref(n_paths), vive_tracker_paths)
                        if xr.check_result(result).is_exception():
                            raise result
                        found_tracker_count = 0


                        now = time.time()
                        if now - current_ts < 1 / frequency:
                            continue
                        current_ts = now

                        for index, space in enumerate(tracker_action_spaces):

                            # space_location = xr.locate_space(space=space, base_space=context.space, time=frame_state.predicted_display_time)
                            space_location = xr.locate_space(space=space, base_space=context.space, time=time.perf_counter_ns())
                            if space_location.location_flags & xr.SPACE_LOCATION_POSITION_VALID_BIT:
                                if role_strings[index] == 'right_elbow':

                                    position = [
                                                    space_location.pose.position.x,
                                                    space_location.pose.position.y,
                                                    space_location.pose.position.z
                                                ]
                                    orientation = [
                                            space_location.pose.orientation.x,
                                            space_location.pose.orientation.y,
                                            space_location.pose.orientation.z,
                                            space_location.pose.orientation.w,
                                        ]

                                    init_pose_quat = np.concatenate([position, orientation])
                                    transformed_pose_rot = compute_pose_transform(init_pose_quat)
                                    # [DEBUG ac947b] track transformed_pose_rot assignment
                                    with open("debug-ac947b.log", "a") as lf:
                                        import json
                                        lf.write(json.dumps({
                                            "sessionId": "ac947b",
                                            "id": f"log_{time.time()}",
                                            "timestamp": int(time.time() * 1000),
                                            "location": "htc_tracker_phone_collect.py:1023",
                                            "message": "transformed_pose_rot assigned",
                                            "data": {
                                                "role": role_strings[index],
                                                "transformed_pose_rot_shape": str(transformed_pose_rot.shape) if transformed_pose_rot is not None else "None",
                                                "transformed_pose_rot": transformed_pose_rot.tolist() if transformed_pose_rot is not None else None,
                                                "frame_count": frame_count,
                                            },
                                            "runId": "debug-run",
                                            "hypothesisId": "A"
                                        }) + "\n")

                                found_tracker_count += 1


                        # add phone camera and pose trajectory
                        # 从异步队列获取最新的Phone数据（非阻塞）
                        phone_frame = try_get_phone_frame()
                        
                        if phone_frame is None:
                            continue

                        # 如果没有有效的 tracker 数据，跳过这一帧
                        if transformed_pose_rot is None:
                            continue
                        
                        rgb_img = phone_frame['image']

                        tx_p, ty_p, tz_p = phone_frame['position']
                        qw, qx, qy, qz = phone_frame['quaternion']
                        
                        # 更新可视化数据（Tracker世界坐标 + 相对位置）
                        phone_pos = [tx_p, ty_p, tz_p]

                        phone_quat = np.array([qw, qx, qy, qz])  # [w, x, y, z]
                        phone_pose_quat = np.concatenate([phone_pos, phone_quat])
                        phone_pose_rot = apply_phone_transform(phone_pose_quat)

                        
                        htc_pose_rot = transformed_pose_rot
                        # [DEBUG ac947b] Hypothesis A/B diagnostic - track transformed_pose_rot
                        with open("debug-ac947b.log", "a") as lf:
                            import json
                            lf.write(json.dumps({
                                "sessionId": "ac947b",
                                "id": f"log_{time.time()}",
                                "timestamp": int(time.time() * 1000),
                                "location": "htc_tracker_phone_collect.py:1048",
                                "message": "htc_pose_rot vs transformed_pose_rot",
                                "data": {
                                    "transformed_pose_rot_shape": str(transformed_pose_rot.shape) if transformed_pose_rot is not None else "None",
                                    "transformed_pose_rot": transformed_pose_rot.tolist() if transformed_pose_rot is not None else None,
                                    "htc_pose_rot_shape": str(htc_pose_rot.shape) if htc_pose_rot is not None else "None",
                                    "frame_count": frame_count,
                                },
                                "runId": "debug-run",
                                "hypothesisId": "A"
                            }) + "\n")


                        if not calibrating and collecting:
                            calib_phone_pose, calib_phone_quat = compute_htc_relative_to_phone(phone_pose_rot, htc_pose_rot, calib_base_phone_pose_rot, calib_base_htc_pose_rot)
                            calib_phone_pose_quat = np.concatenate([calib_phone_pose, calib_phone_quat])
                            calib_phone_pose_rot = quat_to_rot(calib_phone_pose_quat)
                            vis_phone_pose_rot = calib_phone_pose_rot
                        else:
                            vis_phone_pose_rot = phone_pose_rot
                            calib_phone_pose_quat  = phone_pose_quat
                        

                        # ========== 使用新的DataVisualizer进行可视化 ==========
                        # 准备位置数据：使用相对位置 (x, y, z)

                        # 只有在有有效数据时才更新可视化器

                        if htc_pose_rot is not None and vis_phone_pose_rot is not None:
                            visualizer.update(htc_pose_rot, vis_phone_pose_rot, collecting, calibrating)

                        # 生成可视化图像
                        pos_viz_img = visualizer.draw()
                        target_height = rgb_img.shape[0]
                        aspect_ratio = pos_viz_img.shape[1] / pos_viz_img.shape[0]
                        target_width = int(target_height * aspect_ratio)
                        pos_viz_img = cv2.resize(pos_viz_img, (target_width, target_height))

                        # Combine images: phone RGB | position viz | rotation viz
                        combined_img = np.concatenate([rgb_img, pos_viz_img], axis=1)

                        # Add labels
                        cv2.putText(combined_img, "Phone RGB", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(combined_img, "Position (X-Z, X-Y, Z-Y)", (rgb_img.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                        # Show combined image and capture keyboard input
                        cv2.imshow('Combined View', combined_img)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord(' '):
                            collecting = not collecting
                            if collecting:
                                print("Beginning...")
                            else:
                                print("Paused...")
                        elif key == ord('q') or key == ord('Q'):
                            collecting = False
                            print("collected finished.")
                        elif key == ord('e') or key == ord('E'):
                            collecting = False
                            running = False
                            phone_thread_running = False
                            print("Exiting.")
                        elif key == ord('c') or key == ord('C'):
                            calibrating = not calibrating
                            if calibrating:
                                print("world coordinate system calibration reset, prepare to collect data...")
                            else:
                                calib_base_phone_pose_rot = phone_pose_rot
                                calib_base_htc_pose_rot = htc_pose_rot
                                print("world coordinate system calibration finished, prepare to collect data...")
                            
                        elif key == 27:  # ESC
                            collecting = False
                            running = False
                            phone_thread_running = False
                            data_recorder.stop()
                            print("ESC pressed. Exiting.")
                        elif key == 83:  # Right arrow
                            episode_index = int(episode_index) + 1
                            print(f"switch to {episode_index} episode...")
                        elif key == 81:  # Left arrow
                            episode_index = max(0, int(episode_index) - 1)
                            print(f"switch to {episode_index} episode...")

                        frame_count += 1

                        if collecting:
                            htc_pose = np.concatenate([np.array(position), np.array(orientation)]).astype(np.float32)
                            phone_pose = calib_phone_pose_quat.astype(np.float32)
                            phone_images = {'color': phone_frame['image'], 'timestamp': phone_frame.get('timestamp'), 'step_idx': frame_count}
                            phone_point_clouds = None
                            data_dir = f"data/{args.cap_id}_{formatted_time}/pose_{episode_index}"

                            data_recorder.put(htc_pose, phone_pose, phone_images, phone_point_clouds, data_dir)

                        else:
                            data_recorder.put(None, None, None, None, None)

                        if time.time() - last_print > 1.0:
                            last_print = time.time()
                            status = "collecting" if collecting else "paused"
                            status_color = "\033[92m" if collecting else "\033[91m"
                            reset_color = "\033[0m"

                            if position is not None and orientation is not None:
                                rot = Rotation.from_quat(orientation)
                                print(f"{status_color}Episode {episode_index} --- Time: [{last_print}]---[{status}] Data: {position} {rot.as_euler('xyz', degrees=True)}{reset_color}")
                            else:
                                print(f"{status_color}Episode {episode_index} --- Time: [{last_print}]---[{status}] No tracker data{reset_color}")
                        else:
                            continue
        except Exception as e:
            print(f"❌ Error creating OpenXR context: {e}")
            traceback.print_exc()
            exit(1)
                    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--cap_id", type=int, default=112501) # monthday_times
    parser.add_argument("--fps", type=int, default=24)
    args = parser.parse_args()
    main(args)