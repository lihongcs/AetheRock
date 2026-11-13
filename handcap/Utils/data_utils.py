import av
import os
import time
import datetime
import cv2
from collections import deque
import socket
import numpy as np
from scipy.spatial.transform import Rotation

__all__ = [
    "save_frames_to_mp4_with_av",
    "QuestRobotModule",
    "DataVisualizer",
    "precise_wait",
]


def precise_wait(t_end: float, slack_time: float=0.001, time_func=time.monotonic):
    """Copied from umi.common.precise_sleep"""
    t_start = time_func()
    t_wait = t_end - t_start
    if t_wait > 0:
        t_sleep = t_wait - slack_time
        if t_sleep > 0:
            time.sleep(t_sleep)
        while time_func() < t_end:
            pass
    return



def save_frames_to_mp4_with_av(frames, mp4_path, fps=60):

    h, w, _ = frames[0].shape

    with av.open(str(mp4_path), mode='w') as container:
        stream = container.add_stream('h264', rate=int(fps))
        stream.width = w
        stream.height = h
        stream.pix_fmt = 'yuv420p'

        for i, frame in enumerate(frames):
            av_frame = av.VideoFrame.from_ndarray(frame, format='bgr24')
            packet = stream.encode(av_frame)
            container.mux(packet)

        packet = stream.encode(None)
        container.mux(packet)



# For different robot, just write different QuestRightArmLeapModule classes
class QuestRobotModule:
    def __init__(self, vr_ip, local_ip, pose_cmd_port):

        self.vr_ip = vr_ip
        self.local_ip = local_ip
        self.pose_cmd_port = pose_cmd_port
        # Quest should send WorldFrame as well as wrist pose via UDP
        self.wrist_listener_s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.wrist_listener_s.bind(("", pose_cmd_port))
        self.wrist_listener_s.setblocking(1)
        self.wrist_listener_s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 0)

    def compute_rel_transform(self, pose):
        """
        pose: np.ndarray shape (7,) [x, y, z, qx, qy, qz, qw] in unity frame
        """
        world_frame = self.world_frame.copy()
        world_frame[:3] = np.array([world_frame[0], world_frame[2], world_frame[1]])
        pose[:3] = np.array([pose[0], pose[2], pose[1]])

        Q = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0.0]])
        rot_base = Rotation.from_quat(world_frame[3:]).as_matrix()
        rot = Rotation.from_quat(pose[3:]).as_matrix()
        rel_rot = Rotation.from_matrix(
            Q @ (rot_base.T @ rot) @ Q.T
        )  # Is order correct.
        rel_pos = Rotation.from_matrix(Q @ rot_base.T @ Q.T).apply(
            pose[:3] - world_frame[:3]
        )  # Apply base rotation not relative rotation...
        return rel_pos.tolist() + rel_rot.as_quat().tolist()

    def close(self):
        self.wrist_listener_s.close()


    # World frame marks beginning of a program.
    def receive(self):
        """
        pose: np.ndarray shape (7,) [x, y, z, qx, qy, qz, qw] in unity frame
        """

        data, _ = self.wrist_listener_s.recvfrom(1024)
        data_string = data.decode()
        # now = datetime.datetime.now()
        # print(data_string)
        raw_data_list = data_string.split(",")
        dt_type = raw_data_list[0]

        if dt_type == "WF":
            assert len(raw_data_list) == 8
            data_list = [float(data) for data in raw_data_list[1:]]
            world_frame = np.array(data_list)
            self.world_frame = world_frame
            # self.wf_receive_ts = now.strftime("%Y%m%d_%H%M%S")
            # os.mkdir(f"data/pose_{self.wf_receive_ts}")
            # np.save(f"data/pose_{self.wf_receive_ts}/WorldFrame.npy", world_frame)
            return dt_type, {}
        elif dt_type == "EE":
            assert len(raw_data_list) == 9
            is_recording = raw_data_list[1] == "Y"
            data_list = [float(data) for data in raw_data_list[2:]]
            wrist_tf = np.array(data_list[:7])
            # head_tf = np.array(data_list[7:])
            rel_wrist_posquat = self.compute_rel_transform(wrist_tf)
            # rel_head_posquat = self.compute_rel_transform(head_tf)
            return dt_type, {
                "is_rec": is_recording,
                "wrist": rel_wrist_posquat,
                # "head": rel_head_posquat,
            }
        else:
            raise ValueError("Unknown data received from Quest.")




class DataVisualizer:
    def __init__(self, width=600, height=450):
        # 1. 尺寸减半
        self.w = width
        self.h = height
        self.h_top = int(height * 0.6)
        self.h_bot = height - self.h_top
        
        # 状态记录
        self.was_recording = False
        self.pose_history = [] 
        self.angle_history = deque(maxlen=300) 
        
        # 缓存的静态图片 (用于停止录制后固定显示)
        self.frozen_frame = None
        
        # 最小量程配置
        self.MIN_RANGE_POS = 0.5  
        self.MIN_RANGE_ANGLE = 100 

    def update(self, pose, angle, is_recording):
        # --- 状态机逻辑 ---
        
        # A. [开始录制] 时刻 (Rising Edge)
        if is_recording and not self.was_recording:
            self.pose_history = []     # 清空位姿历史
            self.angle_history.clear() # 清空角度历史
            self.frozen_frame = None   # 清除冻结的图片，准备开始动态绘制
        
        # B. [停止录制] 时刻 (Falling Edge)
        elif not is_recording and self.was_recording:
            # 强制绘制最后一帧并缓存起来
            self.frozen_frame = self._render_canvas()
        
        # C. [录制中]
        if is_recording:
            self.pose_history.append(pose)
            self.angle_history.append(angle)
            
        # D. [未录制]
        # 不做任何数据追加，保持 history 不变(或者为空)，
        # 也不需要实时计算，draw() 会直接返回 frozen_frame
            
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
        
        # 如果没有数据（初始状态），直接返回带网格的空背景即可，或者全黑
        if not self.pose_history and not self.angle_history:
            cv2.putText(canvas, "Ready to Record", (self.w//2 - 60, self.h//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100,100,100), 1)
            return canvas

        # ==================== 上半部分：位姿三视图 ====================
        # 只有当有足够数据时才绘制
        if len(self.pose_history) > 1:
            xs = [p[0] for p in self.pose_history]
            ys = [p[1] for p in self.pose_history]
            zs = [p[2] for p in self.pose_history]
            
            min_x, max_x, rng_x = self._get_range_lims(xs, self.MIN_RANGE_POS)
            min_y, max_y, rng_y = self._get_range_lims(ys, self.MIN_RANGE_POS)
            min_z, max_z, rng_z = self._get_range_lims(zs, self.MIN_RANGE_POS)
            
            sub_w = self.w // 3
            sub_h = self.h_top
            
            def plot_trace(idx, data_h, data_v, min_h, rng_h, min_v, rng_v, label_h, label_v):
                sx, sy = idx * sub_w, 0
                self._draw_grid(canvas, sx, sy, sub_w, sub_h, min_v, min_v + rng_v, f"{label_h}-{label_v}")
                
                pts = []
                for i in range(len(data_h)):
                    px = int(sx + 10 + ((data_h[i] - min_h) / rng_h) * (sub_w - 20))
                    py = int(sy + sub_h - 10 - ((data_v[i] - min_v) / rng_v) * (sub_h - 20))
                    pts.append([px, py])
                
                cv2.polylines(canvas, [np.array(pts)], False, (0, 255, 255), 1) # 线条改细一点适应小窗口
                cv2.circle(canvas, tuple(pts[0]), 3, (0, 255, 0), -1)
                cv2.circle(canvas, tuple(pts[-1]), 3, (0, 0, 255), -1)

            plot_trace(0, xs, zs, min_x, rng_x, min_z, rng_z, "X", "Z")
            plot_trace(1, xs, ys, min_x, rng_x, min_y, rng_y, "X", "Y")
            plot_trace(2, zs, ys, min_z, rng_z, min_y, rng_y, "Z", "Y")

        # ==================== 下半部分：角度时序图 ====================
        if len(self.angle_history) > 1:
            vals = list(self.angle_history)
            min_a, max_a, rng_a = self._get_range_lims(vals, self.MIN_RANGE_ANGLE)
            
            sy = self.h_top
            sh = self.h_bot
            self._draw_grid(canvas, 0, sy, self.w, sh, min_a, max_a, "Angle")
            
            pts_angle = []
            step_w = self.w / len(vals)
            for i, val in enumerate(vals):
                px = int(i * step_w)
                py = int(sy + sh - 10 - ((val - min_a) / rng_a) * (sh - 20))
                pts_angle.append([px, py])

            cv2.polylines(canvas, [np.array(pts_angle)], False, (255, 100, 255), 1)
            cv2.putText(canvas, f"val:{vals[-1]}", (self.w - 80, sy + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 255), 1)

        return canvas

    def _get_range_lims(self, data_arr, min_range_val):
        if not data_arr: return 0, 1, 1
        v_min, v_max = min(data_arr), max(data_arr)
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