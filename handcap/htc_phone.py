# Portions adapted from the pyopenxr vive_tracker.py example.
# Copyright 2021 Christopher Bruns. Licensed under Apache-2.0.
# Modified by the AetheRock project for HandCap tracking.

import os
import xr
import cv2
import time
import msvcrt
import lzfse
import ctypes
import socket
import select
import struct
import json
import traceback
import datetime
import threading
import subprocess
import shutil
import numpy as np
from config import *
from ip_config import RASPI_HOST
from collections import deque
from contextlib import contextmanager
from argparse import ArgumentParser
import scipy.spatial.transform as st
from ctypes import cast, byref, POINTER
from scipy.spatial.transform import Rotation
from Postprocess.umi_utils.pose_util import *
from Sensor.htc_phone_process import HTCPhoneDataRecorder

collecting = False
running = True
eposide_index = 0
_stop_delay_start = None

_button_toggle_lock_until = 0.0
_button_was_pressed = False
_button_event_fired = False  # edge-detect flag: one trigger per physical press


UDP_IP = "0.0.0.0"
UDP_PORT_LOCAL = 0

UDP_PORT = 6785
SERVER_ADDR = (RASPI_HOST, UDP_PORT)
MAX_UDP_PACKET = 65535  # Keep large enough to receive reassembled frames from sender
UDP_RCVBUF_BYTES = 32 * 1024 * 1024
UDP_SNDBUF_BYTES = 1 * 1024 * 1024
UDP_FRAME_TIMEOUT = 0.35
UDP_MAX_IN_FLIGHT_FRAMES = 8
UDP_MAX_DRAIN_PACKETS = 512
UDP_STATS_INTERVAL = 5.0

HOST = '127.0.0.1'
PORT = 1337

MAX_PHONE_BODY_SIZE = 32 * 1024 * 1024
PHONE_RECV_IDLE_TIMEOUT = 10.0
PHONE_STALE_TIMEOUT = 2.0
PHONE_CONNECT_TIMEOUT = 8.0
PHONE_CONNECT_RETRY_INTERVAL = 0.2

def is_windows_elevated():
    if os.name != "nt":
        return False
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def get_windows_openxr_runtime():
    if os.name != "nt":
        return None
    try:
        import winreg
        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SOFTWARE\Khronos\OpenXR\1",
        ) as key:
            return winreg.QueryValueEx(key, "ActiveRuntime")[0]
    except Exception:
        return None


def print_openxr_preflight():
    if os.name != "nt":
        return

    api_layer_path = os.environ.get("XR_API_LAYER_PATH")
    active_runtime = get_windows_openxr_runtime()
    elevated = is_windows_elevated()

    print("[OpenXR Preflight]")
    print(f"  elevated/admin: {elevated}")
    print(f"  ActiveRuntime: {active_runtime or '<not found>'}")
    print(f"  XR_API_LAYER_PATH: {api_layer_path or '<not set>'}")
    if elevated and api_layer_path:
        print(
            "  WARNING: OpenXR loader ignores XR_API_LAYER_PATH in elevated/admin shells. "
            "Run this script from a normal Anaconda Prompt/PowerShell."
        )
    if not active_runtime:
        print(
            "  WARNING: No active OpenXR runtime is registered. "
            "Set SteamVR/VIVE/OpenXR runtime as the active OpenXR runtime first."
        )

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
phone_frame_queue = deque(maxlen=10)  # Only keep latest frame
phone_queue_lock = threading.Lock()
phone_thread_running = True

# UDP transfer data queue for async receiving
udp_frame_queue = deque(maxlen=10)  # Only keep latest frame
udp_queue_lock = threading.Lock()
udp_thread_running = True

# Shared display buffer — updated by receiver thread, read by display loop at any pace.
# Not locked for speed; Python GIL makes assignment atomic for pointer-sized values.
_latest_udp_display = None


def configure_udp_socket(sock):
    """Tune UDP buffers for realtime preview without changing recorder behavior."""
    for opt_name, opt_value in (
        ("SO_RCVBUF", UDP_RCVBUF_BYTES),
        ("SO_SNDBUF", UDP_SNDBUF_BYTES),
    ):
        try:
            opt = socket.SO_RCVBUF if opt_name == "SO_RCVBUF" else socket.SO_SNDBUF
            sock.setsockopt(socket.SOL_SOCKET, opt, opt_value)
        except OSError as e:
            print(f"[UDP] Failed to set {opt_name}={opt_value}: {e}")
    try:
        # Low-delay DSCP/TOS hint; ignored on some systems.
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_TOS, 0x10)
    except OSError:
        pass


def is_loopback_host(host):
    return host in ("127.0.0.1", "localhost", "::1")


def _drain_phone_relay_output(proc):
    if proc.stdout is None:
        return
    for line in iter(proc.stdout.readline, ""):
        line = line.strip()
        if line:
            print(f"[Phone Relay] {line}")


def start_phone_relay(args, local_port, device_port):
    tidevice_bin = args.tidevice_bin
    if shutil.which(tidevice_bin) is None:
        raise RuntimeError(
            f"Cannot find '{tidevice_bin}'. Install tidevice or run "
            f"'tidevice relay {local_port} {device_port}' manually and pass --no_tidevice_relay."
        )

    cmd = [tidevice_bin, "relay", str(local_port), str(device_port)]
    print(f"[Phone Relay] Starting: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    threading.Thread(target=_drain_phone_relay_output, args=(proc,), daemon=True).start()
    return proc


def stop_phone_relay(proc):
    if proc is None or proc.poll() is not None:
        return
    print("[Phone Relay] Stopping tidevice relay...")
    proc.terminate()
    try:
        proc.wait(timeout=3.0)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=3.0)


def connect_phone_socket(host, port, timeout, relay_proc=None):
    deadline = time.time() + timeout
    last_error = None

    while time.time() < deadline:
        if relay_proc is not None and relay_proc.poll() is not None:
            raise RuntimeError(f"tidevice relay exited early with code {relay_proc.returncode}")

        try:
            s = socket.create_connection(
                (host, port),
                timeout=min(1.0, max(0.1, deadline - time.time())),
            )
            return s
        except OSError as e:
            last_error = e
            time.sleep(PHONE_CONNECT_RETRY_INTERVAL)

    raise TimeoutError(f"Timed out connecting to AnySense at {host}:{port}: {last_error}")


def prepare_phone_connection(args, host, port):
    relay_proc = None

    try:
        s = connect_phone_socket(host, port, timeout=1.0)
        print(f"[Phone Relay] {host}:{port} is already reachable; using existing relay/server.")
        return s, relay_proc
    except OSError:
        if args.no_tidevice_relay or not is_loopback_host(host):
            raise

    relay_proc = start_phone_relay(args, local_port=port, device_port=args.phone_device_port)
    try:
        s = connect_phone_socket(
            host,
            port,
            timeout=args.phone_connect_timeout,
            relay_proc=relay_proc,
        )
        print(f"[Phone Relay] Ready on {host}:{port} -> iPhone:{args.phone_device_port}")
        return s, relay_proc
    except Exception:
        stop_phone_relay(relay_proc)
        raise


@contextmanager
def managed_phone_connection(args, host, port):
    relay_proc = None
    sock = None
    try:
        sock, relay_proc = prepare_phone_connection(args, host, port)
        yield sock
    finally:
        global phone_thread_running
        phone_thread_running = False
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass
        stop_phone_relay(relay_proc)


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
    additional_transform_matrix[:3, :3] = Rotation.from_euler('y', 180, degrees=True).as_matrix()
    pose_mat = additional_transform_matrix @ pose_mat

    additional_transform_matrix = np.eye(4)
    additional_R_z = Rotation.from_euler('z', 180, degrees=True).as_matrix()
    additional_R_y = Rotation.from_euler('y', -90, degrees=True).as_matrix()
    transform_mat =   additional_R_z @ additional_R_y
    additional_transform_matrix[:3, :3] = transform_mat
    pose_mat_out = pose_mat @ additional_transform_matrix
    # pose_mat_out = pose_mat
    return mat_to_certain_pose_type(pose_mat_out, pose_type="rotvec")

def recv_exact(sock, n, idle_timeout=PHONE_RECV_IDLE_TIMEOUT):
    data = bytearray()
    last_progress_time = time.time()
    while len(data) < n:
        try:
            chunk = sock.recv(n - len(data))
        except socket.timeout:
            if idle_timeout is not None and time.time() - last_progress_time > idle_timeout:
                raise TimeoutError(f"Timed out while receiving {n} bytes; got {len(data)} bytes")
            continue
        if not chunk:
            raise ConnectionResetError("Connection closed by server")
        data.extend(chunk)
        last_progress_time = time.time()
    return bytes(data)


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
            if body_size > MAX_PHONE_BODY_SIZE:
                raise ValueError(f"Invalid phone body_size={body_size}; stream may be out of sync")

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


            # --- 7. Depth data ---
            depth_color = None
            depth_arr = None

            if depth_size > 0 and len(body) >= offset + depth_size:
                depth_compressed = body[offset:offset + depth_size]
                offset += depth_size

                try:
                    depth_raw = lzfse.decompress(depth_compressed)
                except Exception:
                    depth_raw = depth_compressed  # fallback

                raw_len = len(depth_raw)
                expected = depth_w * depth_h * 4

                if raw_len >= expected:
                    depth_arr = np.frombuffer(depth_raw, np.float32).copy().reshape((depth_h, depth_w))
                else:
                    print(f"Warning: depth raw_len={raw_len}, expected>={expected}")
                    depth_arr = None

                if depth_arr is not None:
                    v = depth_arr.ravel()
                    v = v[v > 0.001]
                    v_min = np.percentile(v, 2) if v.size > 10 else depth_arr.min()
                    v_max = np.percentile(v, 98) if v.size > 10 else depth_arr.max()
                    if v_max > v_min:
                        depth_norm = ((depth_arr - v_min) / (v_max - v_min) * 255).clip(0, 255).astype(np.uint8)
                    else:
                        depth_norm = np.zeros(depth_arr.shape, dtype=np.uint8)
                    depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_TURBO)


            frame_data = {
                'image': rgb_img,
                'depth': depth_color,
                'depth_raw': depth_arr,
                'position': [tx_p, ty_p, tz_p],
                'quaternion': [qw, qx, qy, qz],
                'timestamp': time.time()
            }
            
            with phone_queue_lock:
                phone_frame_queue.append(frame_data)
                
        except socket.timeout:
            continue
        except OSError as e:
            # OSError表示socket已无效（可能被关闭），退出循环
            if phone_thread_running:
                print(f"Phone receiver error: {e}")
            break
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


def udp_receiver_thread(sock):
    """Receive UDP preview frames with multi-frame reassembly and stale-frame cleanup."""
    frame_buffers = {}
    packets_received = 0
    malformed_packets = 0
    frames_decoded = 0
    frames_dropped = 0
    frames_missed = 0
    last_completed_frame_id = None
    last_decode_ms = 0.0
    last_stats_time = time.time()

    print("[UDP Receiver] Thread started (multi-frame low-latency mode)...")

    def prune_stale_buffers(now):
        nonlocal frames_dropped
        stale_ids = [
            frame_id for frame_id, buffer in frame_buffers.items()
            if now - buffer["first_seen"] > UDP_FRAME_TIMEOUT
        ]
        for frame_id in stale_ids:
            frame_buffers.pop(frame_id, None)
        frames_dropped += len(stale_ids)

        while len(frame_buffers) > UDP_MAX_IN_FLIGHT_FRAMES:
            oldest_frame_id = min(frame_buffers, key=lambda fid: frame_buffers[fid]["first_seen"])
            frame_buffers.pop(oldest_frame_id, None)
            frames_dropped += 1

    while udp_thread_running:
        try:
            # Drain a bounded burst to prevent OS-buffer backlog, then decode only the newest complete frame.
            packets = []
            while len(packets) < UDP_MAX_DRAIN_PACKETS:
                readable, _, _ = select.select([sock], [], [], 0.005 if not packets else 0.0)
                if not readable:
                    break
                try:
                    data, addr = sock.recvfrom(MAX_UDP_PACKET)
                    packets.append((data, addr, time.time()))
                except (socket.timeout, BlockingIOError):
                    break
                except OSError as e:
                    if getattr(e, 'winerror', 0) == 10040:
                        malformed_packets += 1
                        continue
                    raise

            now = time.time()
            prune_stale_buffers(now)

            if not packets:
                time.sleep(0.001)
                continue

            latest_complete = None

            for data, addr, recv_time in packets:
                packets_received += 1

                if len(data) < 8:
                    malformed_packets += 1
                    continue

                # Parse header: 4B frame_id + 2B chunk_idx + 2B total_chunks
                frame_id, chunk_idx, total_chunks = struct.unpack("!IHH", data[:8])
                payload = data[8:]

                if total_chunks <= 0 or chunk_idx >= total_chunks or total_chunks > 512:
                    malformed_packets += 1
                    continue

                # Late packets from already completed frames cannot help the latest preview.
                if last_completed_frame_id is not None and frame_id <= last_completed_frame_id:
                    continue

                buffer = frame_buffers.get(frame_id)
                if buffer is None or buffer["total"] != total_chunks:
                    if buffer is not None:
                        frames_dropped += 1
                    buffer = {
                        "chunks": {},
                        "total": total_chunks,
                        "first_seen": recv_time,
                        "last_seen": recv_time,
                    }
                    frame_buffers[frame_id] = buffer

                buffer["last_seen"] = recv_time
                buffer["chunks"].setdefault(chunk_idx, payload)

                if len(buffer["chunks"]) == buffer["total"]:
                    full_data = b"".join(buffer["chunks"][i] for i in range(buffer["total"]))
                    frame_buffers.pop(frame_id, None)

                    if last_completed_frame_id is not None and frame_id > last_completed_frame_id + 1:
                        frames_missed += frame_id - last_completed_frame_id - 1
                    last_completed_frame_id = frame_id

                    if latest_complete is None or frame_id >= latest_complete[0]:
                        latest_complete = (frame_id, full_data, recv_time)

            prune_stale_buffers(time.time())

            if latest_complete is not None:
                frame_id, full_data, packet_complete_time = latest_complete
                try:
                    decode_start = time.time()
                    len_bytes = full_data[:4]
                    meta_len = int.from_bytes(len_bytes, byteorder='big')
                    if meta_len < 0 or meta_len > len(full_data) - 4:
                        malformed_packets += 1
                        continue

                    meta_bytes = full_data[4 : 4 + meta_len]
                    jpeg_bytes = full_data[4 + meta_len:]

                    metadata = json.loads(meta_bytes.decode('utf-8'))
                    metadata["udp_frame_id"] = frame_id
                    sender_sys_time_raw = metadata.get('sys_time', 0)
                    sender_sys_time = metadata.get('sys_time_aligned', sender_sys_time_raw)

                    wrist_jpeg_len = metadata.get('wrist_jpeg_len', 0)
                    tactile_left_len = metadata.get('tactile_left_len', 0)
                    tactile_right_len = metadata.get('tactile_right_len', 0)

                    if wrist_jpeg_len <= 0 or wrist_jpeg_len > len(jpeg_bytes):
                        malformed_packets += 1
                        continue

                    left_start = wrist_jpeg_len
                    left_end = left_start + tactile_left_len
                    right_end = left_end + tactile_right_len
                    if left_end > len(jpeg_bytes) or right_end > len(jpeg_bytes):
                        malformed_packets += 1
                        continue

                    wrist_jpeg = jpeg_bytes[:wrist_jpeg_len]
                    left_jpeg = jpeg_bytes[left_start:left_end]
                    right_jpeg = jpeg_bytes[left_end:right_end]

                    wrist_frame = cv2.imdecode(np.frombuffer(wrist_jpeg, np.uint8), cv2.IMREAD_COLOR)
                    if wrist_frame is None:
                        malformed_packets += 1
                        continue

                    tactile_left_frame = cv2.imdecode(np.frombuffer(left_jpeg, np.uint8), cv2.IMREAD_COLOR) \
                        if left_jpeg else None
                    tactile_right_frame = cv2.imdecode(np.frombuffer(right_jpeg, np.uint8), cv2.IMREAD_COLOR) \
                        if right_jpeg else None

                    if tactile_left_frame is None:
                        tactile_left_frame = np.zeros((112, 112, 3), dtype=np.uint8)
                    if tactile_right_frame is None:
                        tactile_right_frame = np.zeros((112, 112, 3), dtype=np.uint8)

                    angle_dict = metadata.get("angle") or {}
                    force_dict = metadata.get("force") or {}
                    gripper = 1.0 if angle_dict.get("status", False) else 0.0
                    left_force = 1.0 if force_dict.get("left", {}).get("status", False) else 0.0
                    right_force = 1.0 if force_dict.get("right", {}).get("status", False) else 0.0
                    button_event = metadata.get("button_event", False)
                    decode_done = time.time()
                    last_decode_ms = (decode_done - decode_start) * 1000

                    global _latest_udp_display
                    with udp_queue_lock:
                        udp_frame_queue.append({
                            'frame_id': frame_id,
                            'wrist_frame': wrist_frame,
                            'tactile_left': tactile_left_frame,
                            'tactile_right': tactile_right_frame,
                            'left_force': left_force,
                            'right_force': right_force,
                            'gripper': gripper,
                            'button_event': button_event,
                            'metadata': metadata,
                            't_recv': packet_complete_time,
                            't_decoded': decode_done,
                            'sender_sys': sender_sys_time,
                            'sender_sys_raw': sender_sys_time_raw,
                            'clock_sync_applied': bool(metadata.get('clock_sync_applied', False)),
                            'clock_sync_offset': float(metadata.get('clock_sync_offset', 0.0)),
                        })
                        _latest_udp_display = udp_frame_queue[-1]

                    frames_decoded += 1

                except Exception as e:
                    print(f"UDP Decode Error: {e}")
                    continue

            now = time.time()
            if now - last_stats_time >= UDP_STATS_INTERVAL:
                with udp_queue_lock:
                    decoded_queue_len = len(udp_frame_queue)
                print(
                    "[UDP Receiver] "
                    f"packets={packets_received} decoded={frames_decoded} "
                    f"missed={frames_missed} dropped_partial={frames_dropped} "
                    f"malformed={malformed_packets} in_flight={len(frame_buffers)} "
                    f"decoded_q={decoded_queue_len} last_decode={last_decode_ms:.1f}ms"
                )
                packets_received = 0
                malformed_packets = 0
                frames_decoded = 0
                frames_dropped = 0
                frames_missed = 0
                last_stats_time = now

        except OSError as e:
            if getattr(e, 'winerror', 0) == 10040:
                continue
            if udp_thread_running:
                print(f"[UDP Receiver] OSError {getattr(e, 'winerror', 0)}: {e}")
            break
        except Exception as e:
            if udp_thread_running:
                import traceback
                traceback.print_exc()
                print(f"[UDP Receiver] error: {e}")
            time.sleep(0.1)
            continue

def try_get_udp_frame():
    """Pop the latest decoded UDP frame (for data recording — timing-critical path)."""
    global _latest_udp_display
    with udp_queue_lock:
        if not udp_frame_queue:
            return None
        frame = udp_frame_queue[-1]
        udp_frame_queue.clear()
        _latest_udp_display = frame          # feed the independent display buffer
        return frame

def try_get_udp_display_frame():
    """Get latest decoded UDP frame for display — no timing dependency on VR loop."""
    return _latest_udp_display

def keyboard_listener():
    global data_recorder, collecting, running
    while running:
        if msvcrt.kbhit():
            key = msvcrt.getch()
            if key == b'\x1b':  # ESC
                collecting = False
                running = False
                data_recorder.stop()
                print("ESC pressed. Exiting.")
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
    if not RASPI_HOST:
        raise SystemExit("Set HANDCAP_RASPI_HOST to the IP address or hostname of the HandCap collector.")

    script_start_time = time.time()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    configure_udp_socket(sock)
    sock.bind((UDP_IP, UDP_PORT_LOCAL))
    sock.settimeout(2.0)
    # Flush any stale packets from the buffer (robust against oversized packets)
    sock.setblocking(False)
    try:
        while True:
            try:
                sock.recv(65535)
            except BlockingIOError:
                break
            except OSError as e:
                if getattr(e, 'winerror', 0) == 10040:
                    break  # oversized stale packet, stop flushing
                raise
    finally:
        sock.setblocking(True)

    print(f"PC UDP Client started. Waiting for server at {SERVER_ADDR}...")
    
    # 双向握手：等待RasPi的HELLO_REPLY
    handshake_timeout = 10.0
    handshake_start = time.time()
    handshake_ok = False
    
    clock_sync_client_minus_server = None
    while time.time() - handshake_start < handshake_timeout:
        # 发送HELLO请求，同时带上本机时间用于立即估计发送端/接收端时钟差。
        client_tx_time = time.time()
        hello_msg = {
            "type": "HELLO_SYNC",
            "client_time": client_tx_time,
        }
        sock.sendto(json.dumps(hello_msg, separators=(",", ":")).encode("utf-8"), SERVER_ADDR)
        print(f"[{time.time() - handshake_start:.1f}s] Sent HELLO_SYNC, waiting for reply...")
        
        try:
            data, addr = sock.recvfrom(65535) 
            print(f"Received data: {data[:50] if len(data) > 50 else data}")
            reply_msg = None
            if data != b"HELLO_REPLY":
                try:
                    reply_msg = json.loads(data.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    reply_msg = None

            if isinstance(reply_msg, dict) and reply_msg.get("type") == "HELLO_REPLY_SYNC":
                client_rx_time = time.time()
                t0 = float(reply_msg.get("client_time", client_tx_time))
                t1 = float(reply_msg.get("server_rx_time", 0.0))
                t2 = float(reply_msg.get("server_tx_time", t1))
                # NTP-style two-way estimate:
                # server_minus_client = ((server_rx - client_tx) + (server_tx - client_rx)) / 2
                server_minus_client = ((t1 - t0) + (t2 - client_rx_time)) * 0.5
                clock_sync_client_minus_server = -server_minus_client
                print(
                    f"✅ Server handshake OK! Connected from {addr}; "
                    f"clock offset client-server={clock_sync_client_minus_server:.6f}s"
                )
                handshake_ok = True
                break
            elif data == b"HELLO_REPLY":
                print(f"✅ Server handshake OK! Connected from {addr}")
                handshake_ok = True
                break
            else:
                print(f"⚠️ Unexpected message, ignoring...")
        except socket.timeout:
            print("Timeout, retrying...")
            continue
        except ConnectionResetError:
            print("Connection reset, RasPi may not be running. Retrying...")
            time.sleep(1)
            continue
        except OSError as e:
            print(f"OSError: {e}, retrying...")
            time.sleep(1)
            continue
    
    if not handshake_ok:
        print("❌ Server handshake failed! Server may not be running.")
        print("   Please run raspi_udp.py on RasPi first!")
        sock.close()
        exit(1)
    
    if clock_sync_client_minus_server is not None:
        ack_msg = {
            "type": "ACK_SYNC",
            "client_minus_server_offset": clock_sync_client_minus_server,
            "client_time": time.time(),
        }
        sock.sendto(json.dumps(ack_msg, separators=(",", ":")).encode("utf-8"), SERVER_ADDR)
        try_get_udp_frame._handshake_clock_offset = clock_sync_client_minus_server
        try_get_udp_frame._clock_offset = clock_sync_client_minus_server
    else:
        sock.sendto(b"ACK", SERVER_ADDR)
    print("🚀 UDP handshake complete! Starting main loop...")

    global collecting, running, eposide_index, data_recorder, _button_toggle_lock_until, _button_event_fired
    local_step_idx = 0
    total_collection_duration = 0.0
    recording_start_time = None
    epoch_frames = {}

    data_recorder = HTCPhoneDataRecorder(save_dir = f"data/htc_phone_{args.cap_id}",
                                        fps = args.fps,
                                        chunksize = 150,
                                        camera_resolution = (960, 720))
    data_recorder.start()

    # 可视化数据

    position = None
    orientation = None

    # Load calibration data
    calib_file = "phone_htc_calib.json"
    if not os.path.exists(calib_file):
        print(f"❌ Calibration file {calib_file} not found! Please run phone_htc_calib.py first.")
        exit(1)
        
    with open(calib_file, "r") as f:
        calib_data = json.load(f)
        calib_base_phone_pose_rot = np.array(calib_data["calib_base_phone_pose_rot"])
        calib_base_htc_pose_rot = np.array(calib_data["calib_base_htc_pose_rot"])
    print(f"✅ Loaded calibration parameters from {calib_file}!")

    print("******** Control Instructions *********************")
    print("******** Press 'ESC' to quit ************************")
    keyboard_thread = threading.Thread(target=keyboard_listener, daemon=True)
    keyboard_thread.start()
    
    phone_host = args.phone_host
    phone_port = args.phone_port
    print(f"Connecting to AnySense at {phone_host}:{phone_port}...")
    if args.no_tidevice_relay or not is_loopback_host(phone_host):
        print("Phone relay auto-start disabled; make sure the AnySense endpoint is reachable.")
    else:
        print(f"Phone relay auto-start enabled: tidevice relay {phone_port} {args.phone_device_port}")

    # UDP transfer image placeholder (will be updated when RasPi sends data)
    udp_transfer_img = None
    _button_event_fired = False

    with managed_phone_connection(args, phone_host, phone_port) as s:
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
                if body_size > MAX_PHONE_BODY_SIZE:
                    raise ValueError(f"Invalid phone body_size={body_size}; stream may be out of sync")
                
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

        global udp_thread_running
        udp_thread_running = True
        udp_thread = threading.Thread(target=udp_receiver_thread, args=(sock,), daemon=True)
        udp_thread.start()
        print("UDP receiver thread started!")

        print_openxr_preflight()
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
                print("Pre-enumerating tracker paths and building action spaces once...")
                _n_paths = ctypes.c_uint32(0)
                result = enumerateViveTrackerPathsHTCX(instance, 0, byref(_n_paths), None)
                if xr.check_result(result).is_exception():
                    raise result
                _vive_tracker_paths = (xr.ViveTrackerPathsHTCX * _n_paths.value)(*([xr.ViveTrackerPathsHTCX()] * _n_paths.value))
                result = enumerateViveTrackerPathsHTCX(instance, _n_paths, byref(_n_paths), _vive_tracker_paths)
                if xr.check_result(result).is_exception():
                    raise result
                _tracker_action_spaces = (xr.Space * len(_vive_tracker_paths))(
                    *[xr.create_action_space(session=session, create_info=xr.ActionSpaceCreateInfo(
                        action=pose_action, subaction_path=tp.paths.principal_path))
                        for tp in _vive_tracker_paths]
                )
                # Index of 'right_elbow' role in role_strings
                _RIGHT_ELBOW_IDX = role_strings.index('right_elbow') if 'right_elbow' in role_strings else -1
                print("Starting main loop - entering frame_loop()...")
                print("NOTE: If session doesn't enter FOCUSED state, you may need to:")
                print("   1. Start a VR app that uses OpenXR")
                print("   2. Or check if Vive Tracker is properly connected")
                print("   3. Ensure SteamVR or other OpenXR runtime is running")

                # Debug: Wait briefly and check session state before entering loop
                print(f"Pre-check: session_state={context.session_state}, session_is_running={context.session_is_running}")

                # Initialize a flag to track first iteration
                first_loop_iteration = True
                last_print = time.time()
                last_healthy_sensor_time = time.time()
                last_phone_stale_log = 0.0
                
                try:
                    from ctypes import windll
                    hwnd = windll.user32.FindWindowW(None, "Combined View")
                    if hwnd:
                        windll.user32.SetWindowPos(hwnd, -1, 0, 0, 0, 0, 0x0001 | 0x0002)
                except Exception:
                    pass

                try:
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

                        if first_loop_iteration:
                            print(f"Session state: {context.session_state}, processing frame...")

                        if context.session_state == xr.SessionState.FOCUSED:
                            active_action_set = xr.ActiveActionSet(action_set=context.default_action_set, subaction_path=xr.NULL_PATH)
                            xr.sync_actions(session=session, sync_info=xr.ActionsSyncInfo(count_active_action_sets=1, active_action_sets=ctypes.pointer(active_action_set)))

                            
                            found_tracker_count = 0
                            cv2.waitKey(1)

                            for index, space in enumerate(tracker_action_spaces):
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

                                    found_tracker_count += 1

                            # ── UDP display: always show latest frame (independent of VR/phone pace) ──
                            udp_frame = try_get_udp_display_frame()

                            if udp_frame is not None:
                                collection_state = udp_frame['metadata'].get('collection_state', 0)
                                new_ep_idx = udp_frame['metadata'].get('episode_index', 0)
                                
                                is_recording_now = (collection_state == 1)
                                
                                global _stop_delay_start
                                if is_recording_now:
                                    if not collecting:
                                        print(f"\n--- Episode {new_ep_idx} Recording Started! ---")
                                        eposide_index = new_ep_idx
                                        local_step_idx = 0
                                        recording_start_time = time.time()
                                        collecting = True
                                    _stop_delay_start = None
                                else:
                                    if collecting:
                                        if _stop_delay_start is None:
                                            _stop_delay_start = time.time()
                                        elif time.time() - _stop_delay_start > 0.5:
                                            print(f"\n--- Episode {eposide_index} Recording Stopped! ---")
                                            if recording_start_time is not None:
                                                total_collection_duration += (time.time() - recording_start_time)
                                                recording_start_time = None
                                            collecting = False
                                            _stop_delay_start = None

                            phone_frame = try_get_phone_frame()
                            if phone_frame is None:
                                phone_frame_available = False
                                if time.time() - last_phone_stale_log > 1.0:
                                    print("[PHONE] No frame received from AnySense yet.")
                                    last_phone_stale_log = time.time()
                                continue
                            else:
                                phone_age = time.time() - phone_frame.get('timestamp', 0)
                                phone_frame_available = phone_age <= PHONE_STALE_TIMEOUT
                                if not phone_frame_available and time.time() - last_phone_stale_log > 1.0:
                                    print(f"[PHONE] Stale AnySense frame: age={phone_age:.2f}s. Receiver may be disconnected.")
                                    last_phone_stale_log = time.time()
                            if transformed_pose_rot is None:
                                tracker_available = False
                            else:
                                tracker_available = True
                            skip_recording = not phone_frame_available or not tracker_available
                            
                            rgb_img = phone_frame['image']
                            depth_img = phone_frame['depth']
                            depth_raw = phone_frame['depth_raw']

                            tx_p, ty_p, tz_p = phone_frame['position']
                            qw, qx, qy, qz = phone_frame['quaternion']
                            
                            # 更新可视化数据（Tracker世界坐标 + 相对位置）
                            phone_pos = [tx_p, ty_p, tz_p]

                            phone_quat = np.array([qw, qx, qy, qz])  # [w, x, y, z]
                            phone_pose_quat = np.concatenate([phone_pos, phone_quat])
                            phone_pose_rot = apply_phone_transform(phone_pose_quat)

                            # add calibrate difference
                            phone_pose_rot = phone_pose_rot.copy()
                            phone_pose_rot[2] += 0.078

                            # 接收同步控制
                            htc_pose_rot = transformed_pose_rot

                            calib_phone_pose, calib_phone_quat = compute_htc_relative_to_phone(phone_pose_rot, htc_pose_rot, calib_base_phone_pose_rot, calib_base_htc_pose_rot)
                            calib_phone_pose_quat = np.concatenate([calib_phone_pose, calib_phone_quat])
                            
                            # ========== 从异步队列获取UDP数据（不阻塞主循环）==========

                            status_color = "\033[92m" if collecting else "\033[91m"

                            # ── STAGE 5: image composition ───────────────────────────────────
                            t_comp_start = time.time()

                            rgb_img = cv2.rotate(rgb_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                            depth_img = cv2.rotate(depth_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                            depth_img_resized = cv2.resize(depth_img, (rgb_img.shape[1], rgb_img.shape[0]))
                            phone_imgs_stacked = np.concatenate([rgb_img, depth_img_resized], axis=0)

                            if udp_frame is not None:
                                wrist_frame     = udp_frame['wrist_frame']
                                tactile_rows = np.hstack([udp_frame['tactile_left'], udp_frame['tactile_right']])
                                udp_transfer_img = np.vstack([wrist_frame, tactile_rows])
                                target_h = phone_imgs_stacked.shape[0]
                                target_w = phone_imgs_stacked.shape[1]
                                src_h, src_w = udp_transfer_img.shape[:2]
                                scale = min(target_w / src_w, target_h / src_h)
                                new_w, new_h = int(src_w * scale), int(src_h * scale)
                                resized = cv2.resize(udp_transfer_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                                canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
                                x_offset = (target_w - new_w) // 2
                                y_offset = (target_h - new_h) // 2
                                canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
                                udp_transfer_img = canvas

                            if udp_transfer_img is None:
                                udp_transfer_img = np.zeros((phone_imgs_stacked.shape[0], phone_imgs_stacked.shape[1], 3), dtype=np.uint8)
                                cv2.putText(udp_transfer_img, "Waiting for UDP...", (20, phone_imgs_stacked.shape[0] // 2),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                            combined_img = np.concatenate([udp_transfer_img, phone_imgs_stacked], axis=1)
                            t_comp_end = time.time()

                            # ── STAGE 6: display ───────────────────────────────────────────
                            t_display_start = time.time()

                            combined_h = 960
                            combined_w = int(combined_h * combined_img.shape[1] / combined_img.shape[0])
                            combined_img = cv2.resize(combined_img, (combined_w, combined_h), interpolation=cv2.INTER_AREA)
                            
                            scale_factor = combined_h / (phone_imgs_stacked.shape[0] or 1)
                            font_scale = max(0.5, 0.7 * scale_factor * 2)
                            thickness = max(1, int(2 * scale_factor * 2))

                            # Add labels
                            cv2.putText(combined_img, "UDP Transfer", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)
                            cv2.putText(combined_img, "Phone RGB", (combined_w // 2 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)
                            cv2.putText(combined_img, "Phone Depth", (combined_w // 2 + 10, combined_h // 2 + 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)

                            # ── Latency overlay (clock-offset corrected) ──────────────────
                            if udp_frame is not None:
                                sender_sys = udp_frame.get('sender_sys', 0)
                                t_recv = udp_frame.get('t_recv', 0)
                                if sender_sys > 0 and t_recv > 0:
                                    if udp_frame.get('clock_sync_applied', False):
                                        clock_offset = 0.0
                                        sync_text = f"Sync:{udp_frame.get('clock_sync_offset', 0.0):.1f}s"
                                    else:
                                        # Correct for sender/receiver clock skew. Prefer handshake estimate;
                                        # fall back to first-frame calibration for older senders.
                                        if not hasattr(try_get_udp_frame, '_clock_offset'):
                                            try_get_udp_frame._clock_offset = getattr(
                                                try_get_udp_frame,
                                                '_handshake_clock_offset',
                                                t_recv - sender_sys,
                                            )
                                        clock_offset = try_get_udp_frame._clock_offset
                                        sync_text = f"Clk:{clock_offset:.1f}s"
                                    end_to_end_ms = (t_recv - sender_sys - clock_offset) * 1000
                                    queue_lat_ms  = (time.time() - t_recv) * 1000
                                    latency_text = f"E2E: {end_to_end_ms:.0f}ms  Q+Dec: {queue_lat_ms:.0f}ms  {sync_text}"
                                    cv2.putText(combined_img, latency_text, (10, combined_h - 10),
                                                cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, (0, 255, 255), thickness)

                            # ── State border and info overlay ──────────────────────────────────────
                            current_duration = total_collection_duration
                            if collecting and recording_start_time is not None:
                                current_duration += (time.time() - recording_start_time)
                            
                            duration_mins = int(current_duration // 60)
                            duration_secs = int(current_duration % 60)

                            if collecting:
                                border_color = (0, 255, 0)    # Green
                                state_text = f"RECORDING | Ep: {eposide_index:04d} | Step: {local_step_idx} | Total: {duration_mins:02d}m{duration_secs:02d}s"
                            else:
                                border_color = (0, 255, 255)  # Yellow
                                state_text = f"PAUSED | Ep: {eposide_index:04d} | Step: {local_step_idx} | Total: {duration_mins:02d}m{duration_secs:02d}s"

                            # Draw state text at top-left with a simple background
                            font_scale_state = font_scale * 1.2
                            bg_y0 = 0
                            bg_y1 = 65
                            cv2.rectangle(combined_img, (0, bg_y0), (combined_w, bg_y1), (20, 20, 20), -1)
                            cv2.putText(combined_img, state_text, (10, 45),
                                        cv2.FONT_HERSHEY_SIMPLEX, font_scale_state, border_color, thickness * 2)


                            cv2.rectangle(combined_img, (0, 0), (combined_w - 1, combined_h - 1), border_color, 40)

                            sensors_ok = True
                            if udp_frame is None:
                                sensors_ok = False
                            else:
                                t_recv = udp_frame.get('t_recv', 0)
                                if time.time() - t_recv > 0.5:
                                    sensors_ok = False
                                else:
                                    meta = udp_frame.get('metadata', {})
                                    if not meta.get('angle', {}).get('status', False):
                                        sensors_ok = False
                                    forces = meta.get('force', {})
                                    for f_v in forces.values():
                                        if not isinstance(f_v, dict) or not f_v.get('status', False):
                                            sensors_ok = False
                                    
                                    if udp_frame.get('tactile_left') is None or not udp_frame['tactile_left'].any():
                                        sensors_ok = False
                                    if udp_frame.get('tactile_right') is None or not udp_frame['tactile_right'].any():
                                        sensors_ok = False
                                    if udp_frame.get('wrist_frame') is None or not udp_frame['wrist_frame'].any():
                                        sensors_ok = False

                            if sensors_ok:
                                last_healthy_sensor_time = time.time()

                            if collecting and not sensors_ok:
                                cv2.rectangle(combined_img, (40, 40), (combined_w - 41, combined_h - 41), (0, 0, 255), 20)
                                if time.time() - last_healthy_sensor_time > 1.0:
                                    alert_text = "INPUT ABNORMAL! CHECK SENSORS!"
                                    text_size = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 5)[0]
                                    text_x = (combined_w - text_size[0]) // 2
                                    text_y = (combined_h + text_size[1]) // 2
                                    overlay = combined_img.copy()
                                    cv2.rectangle(overlay, (0, 0), (combined_w, combined_h), (0, 0, 200), -1)
                                    combined_img = cv2.addWeighted(overlay, 0.3, combined_img, 0.7, 0)
                                    cv2.putText(combined_img, alert_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 5)
                            # Show combined image and capture keyboard input
                            cv2.imshow('Combined View', combined_img)
                            key = cv2.waitKey(1) & 0xFF
                            t_display_end = time.time()

                            t_comp_ms     = (t_comp_end   - t_comp_start) * 1000
                            t_display_ms  = (t_display_end - t_display_start) * 1000

                            if key == 27:  # ESC
                                collecting = False
                                running = False
                                phone_thread_running = False
                                udp_thread_running = False
                                data_recorder.stop()
                                print("ESC pressed. Exiting.")
                                print(f"\n--- Recording Summary ---")
                                print(f"Total Time: {duration_mins:02d}m{duration_secs:02d}s")
                                print("Frames per Epoch:")
                                for ep, frames in sorted(epoch_frames.items()):
                                    print(f"  Ep {ep}: {frames} frames")
                                print("-------------------------\n")
                                # Save to JSON document
                                summary_dict = {
                                    "total_duration_display": f"{duration_mins:02d}m{duration_secs:02d}s",
                                    "total_duration_seconds": current_duration,
                                    "epoch_frames": epoch_frames
                                }
                                summary_path = os.path.join(f"data/htc_phone_{args.cap_id}", "recording_summary.json")
                                os.makedirs(os.path.dirname(summary_path), exist_ok=True)
                                with open(summary_path, 'w') as f:
                                    json.dump(summary_dict, f, indent=4)
                                print(f"✅ Recording summary saved to: {summary_path}")


                            if collecting:
                                if not skip_recording:
                                    htc_pose = np.concatenate([np.array(position), np.array(orientation)]).astype(np.float32)
                                    phone_pose = calib_phone_pose_quat.astype(np.float32)
                                    phone_images = {'color': phone_frame['image'], 'timestamp': phone_frame.get('timestamp'), 'step_idx': local_step_idx}
                                    data_dir = f"data/htc_phone_{args.cap_id}"
                                    depth_dict = {'depth': depth_raw, 'step_idx': local_step_idx, 'timestamp': time.time()} if depth_raw is not None else None
                                    data_recorder.put(htc_pose, phone_pose, phone_images, depth_dict, data_dir, step=local_step_idx)
                                    local_step_idx += 1
                                else:
                                    data_recorder.put(None, None, None, None, None, step=local_step_idx)
                                epoch_frames[eposide_index] = local_step_idx

                            if time.time() - last_print > 1.0:
                                last_print = time.time()
                                status = "collecting" if collecting else "paused"
                                reset_color = "\033[0m"

                                with udp_queue_lock:
                                    qsize = len(udp_frame_queue)
                                udp_status = "UDP:ON" if qsize > 0 else "UDP:OFF"
                                if position is not None and orientation is not None:
                                    rot = Rotation.from_quat(orientation)
                                    print(f"{status_color}[{udp_status}] --- Time: [{last_print}]---[{status}] Data: {position} {rot.as_euler('xyz', degrees=True)}{reset_color}")
                                else:
                                    print(f"{status_color}[{udp_status}] --- Time: [{last_print}]---[{status}] No tracker data{reset_color}")
                            else:
                                continue
                except KeyboardInterrupt:
                    print("\n[Main] Ctrl-C caught. Exiting gracefully...")
                    if collecting and recording_start_time is not None:
                        total_collection_duration += (time.time() - recording_start_time)
                        recording_start_time = None
                    collecting = False
                    running = False
                    phone_thread_running = False
                    udp_thread_running = False
                    if 'data_recorder' in globals():
                        data_recorder.stop()
        except Exception as e:
            print(f"❌ Error creating OpenXR context: {e}")
            if os.name == "nt":
                print("\n[OpenXR 修复建议]")
                print("  1. 不要用管理员身份运行 Anaconda Prompt/PowerShell/IDE。关闭当前窗口，重新开一个普通窗口。")
                print("  2. 确认 SteamVR/VIVE Console/VIVE Business Streaming 已启动，头显和 Tracker 都在线。")
                print("  3. 在 SteamVR 设置里把 SteamVR 设为当前 OpenXR Runtime，或在 VIVE Console 里设置对应 runtime。")
                print("  4. 如果刚切换 runtime，重启 SteamVR/VIVE 服务后再运行脚本。")
            traceback.print_exc()
            exit(1)
                    

    if 'script_start_time' in locals():
        total_runtime = time.time() - script_start_time
        def format_time(seconds):
            h = int(seconds // 3600)
            m = int((seconds % 3600) // 60)
            s = int(seconds % 60)
            return f"{h:02d}:{m:02d}:{s:02d}"
            
        try:
            summary_file = os.path.join(save_dir, "collection_summary.txt")
            with open(summary_file, "w") as f:
                f.write(f"Total Runtime (seconds): {total_runtime:.2f}\n")
                f.write(f"Total Runtime (formatted): {format_time(total_runtime)}\n")
                f.write(f"Effective Data Time (seconds): {total_collection_duration:.2f}\n")
                f.write(f"Effective Data Time (formatted): {format_time(total_collection_duration)}\n")
                f.write(f"Total Episodes Recorded: {eposide_index}\n")
            print(f"\n[Summary] Saved time tracking to {summary_file}")
        except Exception as e:
            print(f"\n[Summary] Failed to save time tracking: {e}")
            
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--cap_id", type=int, default=112501) # monthday_times
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--phone_host", "--phone-host", dest="phone_host", type=str, default=HOST)
    parser.add_argument("--phone_port", "--phone-port", dest="phone_port", type=int, default=PORT)
    parser.add_argument("--phone_device_port", "--phone-device-port", dest="phone_device_port", type=int, default=PORT)
    parser.add_argument("--phone_connect_timeout", "--phone-connect-timeout", dest="phone_connect_timeout",
                        type=float, default=PHONE_CONNECT_TIMEOUT)
    parser.add_argument("--tidevice_bin", "--tidevice-bin", dest="tidevice_bin", type=str, default="tidevice")
    parser.add_argument("--no_tidevice_relay", "--no-tidevice-relay", dest="no_tidevice_relay",
                        action="store_true")
    args = parser.parse_args()
    main(args)
