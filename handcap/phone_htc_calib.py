import os
import xr
import cv2
import sys
import time
try:
    import msvcrt
except ImportError:
    msvcrt = None
import lzfse
import ctypes
import socket
import struct
import json
import traceback
import datetime
import threading
import numpy as np
from collections import deque
from ctypes import cast, byref, POINTER
from argparse import ArgumentParser
from scipy.spatial.transform import Rotation
import scipy.spatial.transform as st
from Postprocess.umi_utils.pose_util import pose_to_mat, mat_to_certain_pose_type


running = True

HOST = '127.0.0.1'
PORT = 1337

PEERTALK_HEADER_SIZE = 16
RECORD3D_HEADER_SIZE = 44
INTRINSIC_SIZE = 16
POSE_SIZE = 28


phone_frame_queue = deque(maxlen=10)
phone_queue_lock = threading.Lock()
phone_thread_running = True


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
    sock.settimeout(0.1)
    
    while phone_thread_running:
        try:
            pt_raw = recv_exact(sock, PEERTALK_HEADER_SIZE)
            pt_a, pt_b, pt_c = struct.unpack_from('<III', pt_raw, 0)
            body_size = struct.unpack_from('>I', pt_raw, 12)[0]

            if body_size == 0:
                continue

            body = recv_exact(sock, body_size)
            offset = 0

            if len(body) < offset + RECORD3D_HEADER_SIZE:
                continue
            (rgb_w, rgb_h, depth_w, depth_h, conf_w, conf_h,
                rgb_size, depth_size, conf_size, misc_size, device_type
                ) = struct.unpack_from('<11I', body, offset)
            offset += RECORD3D_HEADER_SIZE

            if len(body) < offset + INTRINSIC_SIZE:
                continue
            fx, fy, tx, ty = struct.unpack_from('<4f', body, offset)
            offset += INTRINSIC_SIZE

            if len(body) < offset + POSE_SIZE:
                continue
            qx, qy, qz, qw, tx_p, ty_p, tz_p = struct.unpack_from('<7f', body, offset)
            offset += POSE_SIZE

            if rgb_size == 0 or len(body) < offset + rgb_size:
                continue

            rgb_jpeg = body[offset:offset + rgb_size]
            offset += rgb_size

            nparr = np.frombuffer(rgb_jpeg, np.uint8)
            rgb_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            depth_color = None
            if depth_size > 0 and len(body) >= offset + depth_size:
                depth_compressed = body[offset:offset + depth_size]
                offset += depth_size

                try:
                    depth_raw = lzfse.decompress(depth_compressed)
                except Exception:
                    depth_raw = depth_compressed

                raw_len = len(depth_raw)
                expected = depth_w * depth_h * 4

                if raw_len >= expected:
                    depth_arr = np.frombuffer(depth_raw, np.float32).copy().reshape((depth_h, depth_w))
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
                'position': [tx_p, ty_p, tz_p],
                'quaternion': [qw, qx, qy, qz],
                'timestamp': time.time()
            }
            
            with phone_queue_lock:
                phone_frame_queue.append(frame_data)
                
        except socket.timeout:
            continue
        except OSError as e:
            if phone_thread_running:
                print(f"Phone receiver error: {e}")
            break
        except Exception as e:
            continue


def try_get_phone_frame():
    with phone_queue_lock:
        if phone_frame_queue:
            return phone_frame_queue[-1]
        return None


class DataVisualizer:
    def __init__(self, width=480, height=960):
        self.w = width
        self.h = height
        self.h_top = int(height * 0.5)
        self.h_bot = height - self.h_top
        self.sub_w = self.w // 4
        self.h_pos = self.h_top

        self.was_recording = False
        self.htc_pose_history = []
        self.phone_pose_history = []
        self.frozen_frame = None

        self._static_bg = None
        self.MIN_RANGE_POS = 0.5
        self.PITCH = np.radians(30)
        self.AXIS_LEN_FACTOR = 0.12

        self._prev_htc_pts  = [None] * 4
        self._prev_phone_pts = [None] * 4
        self._build_static_bg()

    def _build_static_bg(self):
        bg = np.full((self.h, self.w, 3), 15, dtype=np.uint8)

        def draw_grid(x, y, w, h, title):
            cv2.rectangle(bg, (x, y), (x+w, y+h), (40, 40, 40), -1)
            cv2.rectangle(bg, (x, y), (x+w, y+h), (80, 80, 80), 1)
            cv2.putText(bg, title, (x+5, y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        sw, hp = self.sub_w, self.h_pos
        draw_grid(0,      0, sw, hp, "X-Z")
        draw_grid(sw,     0, sw, hp, "X-Y")
        draw_grid(2*sw,   0, sw, hp, "Z-Y")
        draw_grid(3*sw,   0, sw, hp, "3D")
        cv2.putText(bg, "HTC Tracker", (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        
        sy = self.h_top
        draw_grid(0,      sy, sw, hp, "X-Z")
        draw_grid(sw,     sy, sw, hp, "X-Y")
        draw_grid(2*sw,   sy, sw, hp, "Z-Y")
        draw_grid(3*sw,   sy, sw, hp, "3D")
        cv2.putText(bg, "Phone Pose", (10, sy + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 100), 1)
        
        cv2.line(bg, (0, self.h_top), (self.w, self.h_top), (80, 80, 80), 1)
        cv2.line(bg, (3*sw, 0),      (3*sw, self.h_top),  (80, 80, 80), 1)
        cv2.line(bg, (3*sw, self.h_top), (3*sw, self.h), (80, 80, 80), 1)
        self._static_bg = bg

    def _range(self, vals, min_val=0.5):
        v_min, v_max = min(vals), max(vals)
        rng = max(v_max - v_min, min_val)
        if v_max - v_min < min_val:
            center = (v_max + v_min) / 2
            v_min, v_max = center - min_val / 2, center + min_val / 2
        return v_min, v_max, rng

    def _proj(self, x, y, sx, sy, sw, sh,
               x_min, x_rng, y_min, y_rng):
        px = int(sx + 15 + ((x - x_min) / x_rng) * (sw - 30))
        py = int(sy + sh - 15 - ((y - y_min) / y_rng) * (sh - 30))
        return px, py

    def _draw_tiles(self, valid_poses, sy_base, prev_pts, color):
        if len(valid_poses) < 2:
            return [None] * 4

        xs = [p[0] for p in valid_poses]
        ys = [p[1] for p in valid_poses]
        zs = [p[2] for p in valid_poses]
        x_min, x_max, x_rng = self._range(xs, self.MIN_RANGE_POS)
        y_min, y_max, y_rng = self._range(ys, self.MIN_RANGE_POS)
        z_min, z_max, z_rng = self._range(zs, self.MIN_RANGE_POS)

        cos_p = np.cos(self.PITCH)
        sin_p = np.sin(self.PITCH)
        sw = self.sub_w
        sh = self.h_pos

        def proj_xz(i, p): return self._proj(p[0], p[2], 0, sy_base, sw, sh, x_min, x_rng, z_min, z_rng)
        def proj_xy(i, p): return self._proj(p[0], p[1], sw, sy_base, sw, sh, x_min, x_rng, y_min, y_rng)
        def proj_zy(i, p): return self._proj(p[2], p[1], 2*sw, sy_base, sw, sh, z_min, z_rng, y_min, y_rng)

        y_p_min = y_min * cos_p - z_max * sin_p
        y_p_max = y_max * cos_p - z_min * sin_p
        rng_y_proj = max(y_p_max - y_p_min, 1e-9)
        y_proj_ref = y_p_min
        def proj_3d(i, p):
            px = int(3*sw + 15 + ((p[0] - x_min) / x_rng) * (sw - 30))
            y_p = cos_p * p[1] - sin_p * p[2]
            py = int(sy_base + sh - 15 - ((y_p - y_proj_ref) / rng_y_proj) * (sh - 30))
            return px, py

        proj_funcs = [proj_xz, proj_xy, proj_zy, proj_3d]

        n = len(valid_poses)
        last = valid_poses[-1]
        last_rotvec = last[3:6] if len(last) >= 6 else None
        end_pts = [None] * 4
        
        for i in range(1, n):
            p_prev = valid_poses[i - 1]
            p_curr = valid_poses[i]
            for k, proj in enumerate(proj_funcs):
                pt_prev = proj(i - 1, p_prev)
                pt_curr = proj(i, p_curr)
                cv2.line(self._canvas, pt_prev, pt_curr, color, 1)
                
        for k, proj in enumerate(proj_funcs):
            end_pts[k] = proj(n - 1, last)
            
        for pt in end_pts:
            if pt is not None:
                cv2.circle(self._canvas, pt, 4, (0, 0, 255), -1)

        if last_rotvec is not None:
            end_3d = end_pts[3]
            try:
                rv_arr = np.asarray(last_rotvec)
                rot = st.Rotation.from_rotvec(rv_arr)
                R = rot.as_matrix()
                axis_len = int(sh * self.AXIS_LEN_FACTOR)
                denom = x_rng + y_rng + z_rng + 1e-9
                for col, bgr in [(R[:, 0], (0, 0, 255)), (R[:, 1], (0, 255, 0)), (R[:, 2], (255, 0, 0))]:
                    dx, dy, dz = col
                    ax = int(dx * axis_len * x_rng / denom * 3)
                    ay = int((cos_p * dy - sin_p * dz) * axis_len / denom * 3)
                    cv2.arrowedLine(self._canvas, end_3d, (end_3d[0] + ax, end_3d[1] - ay), bgr, 2, tipLength=0.3)
            except Exception:
                pass

        return end_pts

    def draw(self):
        canvas = self._static_bg.copy()
        self._canvas = canvas

        htc_valid = [p for p in self.htc_pose_history if p is not None]
        phone_valid = [p for p in self.phone_pose_history if p is not None]

        if len(htc_valid) > 1:
            self._prev_htc_pts = self._draw_tiles(htc_valid, 0, self._prev_htc_pts, (0, 255, 255))
        if len(phone_valid) > 1:
            self._prev_phone_pts = self._draw_tiles(phone_valid, self.h_top, self._prev_phone_pts, (255, 200, 100))

        self._canvas = None
        return canvas

    def update(self, htc_pose, phone_pose):
        self.htc_pose_history.append(htc_pose)
        self.phone_pose_history.append(phone_pose)
        if len(self.htc_pose_history) > 300:
            self.htc_pose_history = self.htc_pose_history[-300:]
        if len(self.phone_pose_history) > 300:
            self.phone_pose_history = self.phone_pose_history[-300:]


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
        self.swapchain_image_buffers = []
        self.exit_render_loop = False
        self.request_restart = False
        self.session_is_running = False

    def __enter__(self):
        self.instance = xr.create_instance(create_info=self._instance_create_info)
        self.system_id = xr.get_system(instance=self.instance, get_info=xr.SystemGetInfo(form_factor=self.form_factor))
        if self._session_create_info.next is not None:
            self.graphics_binding_pointer = self._session_create_info.next
        self._session_create_info.system_id = self.system_id
        self.session = xr.create_session(instance=self.instance, create_info=self._session_create_info)
        self.space = xr.create_reference_space(session=self.session, create_info=self._reference_space_create_info)
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
        if self.space is not None:
            xr.destroy_space(self.space)
        if self.session is not None:
            xr.destroy_session(self.session)
        if self.graphics is not None:
            self.graphics.destroy()
        if self.instance is not None:
            xr.destroy_instance(self.instance)

    def poll_xr_events(self):
        self.exit_render_loop = False
        while True:
            try:
                event_buffer = xr.poll_event(self.instance)
                try:
                    event_type = xr.StructureType(event_buffer.type)
                except ValueError:
                    continue
                if event_type == xr.StructureType.EVENT_DATA_INSTANCE_LOSS_PENDING:
                    self.exit_render_loop = True
                elif event_type == xr.StructureType.EVENT_DATA_SESSION_STATE_CHANGED and self.session is not None:
                    event = cast(byref(event_buffer), POINTER(xr.EventDataSessionStateChanged)).contents
                    self.session_state = xr.SessionState(event.state)
                    if self.session_state == xr.SessionState.READY:
                        xr.begin_session(session=self.session, begin_info=xr.SessionBeginInfo(self.view_configuration_type))
                        self.session_is_running = True
                    elif self.session_state == xr.SessionState.STOPPING:
                        self.session_is_running = False
                        xr.end_session(self.session)
                    elif self.session_state in (xr.SessionState.EXITING, xr.SessionState.LOSS_PENDING):
                        self.exit_render_loop = True
            except xr.EventUnavailable:
                break

    def frame_loop(self):
        xr.attach_session_action_sets(
            session=self.session,
            attach_info=xr.SessionActionSetsAttachInfo(
                count_action_sets=len(self.action_sets),
                action_sets=(xr.ActionSet * len(self.action_sets))(*self.action_sets)
            ),
        )
        while True:
            self.exit_render_loop = False
            self.poll_xr_events()
            if self.exit_render_loop:
                break
            if self.session_is_running:
                if self.session_state in (xr.SessionState.READY, xr.SessionState.SYNCHRONIZED, xr.SessionState.VISIBLE, xr.SessionState.FOCUSED):
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
                time.sleep(0.250)


def main():
    global running, phone_thread_running
    
    print("******** Phone & HTC Tracker Calibration ***************")
    print("******** Please place Tracker and Phone in positions ***")
    print("******** Press 'c' to save parameters ******************")
    print("******** Press 'ESC' to quit ***************************")

    pos_viz_width = 720
    pos_viz_height = 960
    visualizer = DataVisualizer(width=pos_viz_width, height=pos_viz_height)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        print("✅ Connected! Waiting for phone data (USB mode + recording)...")
        s.settimeout(30.0)
        phone_ready = False
        while not phone_ready:
            try:
                pt_raw = recv_exact(s, PEERTALK_HEADER_SIZE)
                body_size = struct.unpack_from('>I', pt_raw, 12)[0]
                if body_size == 0: continue
                body = recv_exact(s, body_size)
                if len(body) < RECORD3D_HEADER_SIZE: continue
                (rgb_w, rgb_h, depth_w, depth_h, conf_w, conf_h,
                 rgb_size, depth_size, conf_size, misc_size, device_type) = struct.unpack_from('<11I', body, 0)
                if rgb_size > 0 and device_type in [0, 1, 2]:
                    phone_ready = True
                    print("✅ Phone ready! USB mode detected.")
            except socket.timeout:
                print("⏱️ Timeout waiting... Retrying...")
            except Exception as e:
                time.sleep(1)
        
        phone_thread_running = True
        receiver_thread = threading.Thread(target=phone_receiver_thread, args=(s,), daemon=True)
        receiver_thread.start()

        print("Creating OpenXR context...")
        try:
            with ContextObject(
                instance_create_info=xr.InstanceCreateInfo(
                    enabled_extension_names=[xr.MND_HEADLESS_EXTENSION_NAME, "XR_HTCX_vive_tracker_interaction"],
                ),
            ) as context:
                instance = context.instance
                session = context.session

                enumerateViveTrackerPathsHTCX = cast(
                    xr.get_instance_proc_addr(instance, "xrEnumerateViveTrackerPathsHTCX"),
                    xr.PFN_xrEnumerateViveTrackerPathsHTCX
                )

                role_strings = ["handheld_object", "left_foot", "right_foot", "left_shoulder", "right_shoulder",
                                "left_elbow", "right_elbow", "left_knee", "right_knee", "waist", "chest", "camera", "keyboard"]
                role_path_strings = [f"/user/vive_tracker_htcx/role/{role}" for role in role_strings]
                role_paths = (xr.Path * len(role_path_strings))(*[xr.string_to_path(instance, r) for r in role_path_strings])
                
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
                    *[xr.ActionSuggestedBinding(pose_action, xr.string_to_path(instance, f"{r}/input/grip/pose")) for r in role_path_strings]
                )
                
                xr.suggest_interaction_profile_bindings(instance=instance, suggested_bindings=xr.InteractionProfileSuggestedBinding(
                    interaction_profile=xr.string_to_path(instance, "/interaction_profiles/htc/vive_tracker_htcx"),
                    count_suggested_bindings=len(suggested_binding_paths), suggested_bindings=suggested_binding_paths,
                ))
                
                tracker_action_spaces = (xr.Space * len(role_paths))(
                    *[xr.create_action_space(session=session, create_info=xr.ActionSpaceCreateInfo(action=pose_action, subaction_path=r)) for r in role_paths]
                )

                _n_paths = ctypes.c_uint32(0)
                enumerateViveTrackerPathsHTCX(instance, 0, byref(_n_paths), None)
                _vive_tracker_paths = (xr.ViveTrackerPathsHTCX * _n_paths.value)(*([xr.ViveTrackerPathsHTCX()] * _n_paths.value))
                enumerateViveTrackerPathsHTCX(instance, _n_paths, byref(_n_paths), _vive_tracker_paths)

                try:
                    from ctypes import windll
                    hwnd = windll.user32.FindWindowW(None, "Calibration View")
                    if hwnd: windll.user32.SetWindowPos(hwnd, -1, 0, 0, 0, 0, 0x0001 | 0x0002)
                except Exception:
                    pass

                print("Starting visualization loop...")
                for frame_state in context.frame_loop():
                    if not running:
                        break

                    if context.session_state != xr.SessionState.FOCUSED:
                        time.sleep(0.1)
                        continue

                    active_action_set = xr.ActiveActionSet(action_set=context.default_action_set, subaction_path=xr.NULL_PATH)
                    xr.sync_actions(session=session, sync_info=xr.ActionsSyncInfo(count_active_action_sets=1, active_action_sets=ctypes.pointer(active_action_set)))
                    
                    transformed_pose_rot = None
                    for index, space in enumerate(tracker_action_spaces):
                        space_location = xr.locate_space(space=space, base_space=context.space, time=time.perf_counter_ns())
                        if space_location.location_flags & xr.SPACE_LOCATION_POSITION_VALID_BIT:
                            if role_strings[index] == 'right_elbow':
                                position = [space_location.pose.position.x, space_location.pose.position.y, space_location.pose.position.z]
                                orientation = [space_location.pose.orientation.x, space_location.pose.orientation.y, space_location.pose.orientation.z, space_location.pose.orientation.w]
                                init_pose_quat = np.concatenate([position, orientation])
                                transformed_pose_rot = compute_pose_transform(init_pose_quat)
                                break

                    phone_frame = try_get_phone_frame()
                    if phone_frame is None or transformed_pose_rot is None:
                        cv2.waitKey(1)
                        continue

                    rgb_img = phone_frame['image']
                    depth_img = phone_frame['depth']
                    if depth_img is None:
                        depth_img = np.zeros_like(rgb_img)

                    tx_p, ty_p, tz_p = phone_frame['position']
                    qw, qx, qy, qz = phone_frame['quaternion']
                    
                    phone_pos = [tx_p, ty_p, tz_p]
                    phone_quat = np.array([qw, qx, qy, qz])  # [w, x, y, z]
                    phone_pose_quat = np.concatenate([phone_pos, phone_quat])
                    phone_pose_rot = apply_phone_transform(phone_pose_quat)
                    phone_pose_rot[2] += 0.078
                    
                    htc_pose_rot = transformed_pose_rot

                    visualizer.update(htc_pose_rot, phone_pose_rot)
                    pos_viz_img = visualizer.draw()

                    target_w, target_h = rgb_img.shape[1], rgb_img.shape[0]
                    
                    if pos_viz_img.shape[0] != target_h:
                        viz_scale = target_h / pos_viz_img.shape[0]
                        new_w = int(pos_viz_img.shape[1] * viz_scale)
                        pos_viz_img = cv2.resize(pos_viz_img, (new_w, target_h))

                    depth_img_resized = cv2.resize(depth_img, (target_w, target_h))
                    
                    combined_img = np.concatenate([rgb_img, depth_img_resized, pos_viz_img], axis=1)
                    
                    combined_h = 640
                    combined_w = int(combined_h * combined_img.shape[1] / combined_img.shape[0])
                    combined_img = cv2.resize(combined_img, (combined_w, combined_h), interpolation=cv2.INTER_AREA)

                    font_scale = 0.7 * combined_h / target_h
                    thickness = max(1, int(2 * combined_h / target_h))

                    cv2.putText(combined_img, "Phone RGB", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)
                    cv2.putText(combined_img, "Phone Depth", (target_w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)
                    cv2.putText(combined_img, "Position Viewer (HTC & Phone)", (target_w * 2 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)

                    cv2.putText(combined_img, "Press 'c' to SAVE CALIBRATION", (10, combined_h - 20), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 1.5, (0, 0, 255), thickness + 1)
                    
                    cv2.imshow("Calibration View", combined_img)
                    key = cv2.waitKey(1) & 0xFF
                    
                    if msvcrt and msvcrt.kbhit():
                        key_m = msvcrt.getch()
                        if key_m in [b'c', b'C']: key = ord('c')
                        elif key_m == b'\x1b': key = 27

                    if key == ord('c') or key == ord('C'):
                        try:
                            calib_data = {
                                "calib_base_phone_pose_rot": phone_pose_rot.tolist(),
                                "calib_base_htc_pose_rot": htc_pose_rot.tolist()
                            }
                            with open("phone_htc_calib.json", "w") as f:
                                json.dump(calib_data, f, indent=4)
                            print(f"\n✅ SUCCESS! Calibration saved to phone_htc_calib.json\nData: {calib_data}\n")
                            # visual feedback
                            cv2.putText(combined_img, "SAVED!", (combined_w // 2 - 100, combined_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 255, 0), 5)
                            cv2.imshow("Calibration View", combined_img)
                            cv2.waitKey(500)
                        except Exception as e:
                            print(f"Failed to save JSON: {e}")
                    elif key == 27:
                        running = False
                        phone_thread_running = False
                        print("ESC pressed. Exiting.")
                        break

        except Exception as e:
            print(f"❌ Error: {e}")
            traceback.print_exc()
            exit(1)


if __name__ == "__main__":
    main()
