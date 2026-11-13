
import os
import cv2
import json
import time
import math
import queue
import socket
import struct
import threading
import statistics
import numpy as np
from config import *
from ip_config import NTP_HOST
from argparse import ArgumentParser
from Sensor.angle import AngleSensor
from Sensor.mvs_cam import MVSCamera
from multiprocessing import Value, Process
from Sensor.force import (ForceReader, ForceSensor)
from multiprocessing.managers import SharedMemoryManager
from Sensor.camera_process import TactileCamera, MVSWristCamera, TactileUSBCamera
import sys
import atexit
import select
try:
    import msvcrt
except ImportError:
    msvcrt = None

_old_term_settings = None

def init_keyboard():
    global _old_term_settings
    if sys.platform != 'win32' and sys.stdin.isatty():
        import termios
        import tty
        _old_term_settings = termios.tcgetattr(sys.stdin.fileno())
        tty.setcbreak(sys.stdin.fileno())

def restore_keyboard():
    if sys.platform != 'win32' and sys.stdin.isatty() and _old_term_settings is not None:
        import termios
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, _old_term_settings)

atexit.register(restore_keyboard)

def check_esc_pressed():
    if sys.platform == 'win32':
        if msvcrt and msvcrt.kbhit():
            return msvcrt.getch() == b'\x1b'
    else:
        if sys.stdin.isatty():
            import select
            if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
                ch = sys.stdin.read(1)
                return ch == '\x1b'
    return False


# ── UDP / hardware constants ────────────────────────────────────────────────
UDP_IP = "0.0.0.0"
UDP_PORT = 6785  # opened on raspi for sending everything
MAX_PACKET_SIZE = 1400  # 1.4KB packets — avoid IP fragmentation under 1500 MTU
CHUNK_HEADER_SIZE = 8  # 4B frame_id (full 32bit) + 2B chunk_idx + 2B total_chunks
UDP_SNDBUF_BYTES = 4 * 1024 * 1024
UDP_SEND_QUEUE_SIZE = 2
UDP_SEND_BURST_CHUNKS = 8

# ── Async encoding constants ─────────────────────────────────────────────────
ASYNC_QUEUE_SIZE = 3  # UDP preview only: keep latest frames and drop stale frames under load.
ENCODER_THREAD_FPS = 20  # 提升到 20Hz，与数据保存频率统一

# ── Debug Configuration ────────────────────────────────────────────────────────
_DEBUG_VERBOSE = False          # 开启详细调试打印
_DEBUG_FRAME_LOG = False        # 每帧打印一次采集和入队信息
_DEBUG_ENCODE_LOG = False       # 编码详情打印
_DEBUG_SEND_LOG = False          # 发送详情打印
_DEBUG_TIMING_LOG = False        # 时间间隔统计打印
_DEBUG_PACKET_LOG = False       # 每个发送包打印（关闭，太冗长会拖慢发送线程）

RGB_SENSOR_ERROR_DEBOUNCE_SEC = 1.0
RGB_SENSOR_RECOVERY_DEBOUNCE_SEC = 0.5
RGB_LIGHT_REFRESH_SEC = 1.0
NTP_CHECK_SAMPLES = 8
NTP_CHECK_SAMPLE_GAP_SEC = 0.02


def configure_udp_socket(sock):
    """Tune UDP for realtime preview without coupling it to data saving."""
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, UDP_SNDBUF_BYTES)
    except OSError as e:
        print(f"[UDP] Failed to set SO_SNDBUF={UDP_SNDBUF_BYTES}: {e}")
    try:
        # Low-delay DSCP/TOS hint. Some OS/network stacks may ignore it.
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_TOS, 0x10)
    except OSError:
        pass



# ── GPIO Button Process ───────────────────────────────────────────────────────
GPIO_PATH = f"/sys/class/gpio/gpio{GPIO_NUM}"

def gpio_monitor_process(shared_value, value_fd_out, press_time_out, poll_interval=0.01):
    """
    独立进程：持续监控GPIO状态。
    
    shared_value:    当前GPIO电平 (0=按下, 1=松开)
    value_fd_out:    value 文件描述符编号（给主进程用）
    press_time_out:  最近一次按下事件的时间戳（Unix时间，秒）
    poll_interval:   轮询间隔
    """
    if not os.path.exists(GPIO_PATH):
        with open("/sys/class/gpio/export", "w") as f:
            f.write(str(GPIO_NUM))
        time.sleep(0.1)
    
    with open(f"{GPIO_PATH}/direction", "w") as f:
        f.write("in")
    
    value_fd = os.open(f"{GPIO_PATH}/value", os.O_RDONLY)
    
    # 传回 fd 编号
    with value_fd_out.get_lock():
        value_fd_out.value = value_fd
    
    # 初始化按下时间（0表示无按下）
    with press_time_out.get_lock():
        press_time_out.value = 0.0
    
    last_state = int(os.read(value_fd, 1))
    os.lseek(value_fd, 0, os.SEEK_SET)
    
    while True:
        try:
            current_state = int(os.read(value_fd, 1))
            os.lseek(value_fd, 0, os.SEEK_SET)
            
            if current_state != last_state:
                with shared_value.get_lock():
                    shared_value.value = current_state
                
                # ★ 按下时（current_state==0）记录时间戳
                if current_state == 0:
                    with press_time_out.get_lock():
                        press_time_out.value = time.time()
                
                last_state = current_state
            
            time.sleep(poll_interval)
        except Exception as e:
            print(f"[GPIO Monitor] Error: {e}")
            time.sleep(poll_interval)


class GPIOButtonMonitor:
    """GPIO按钮监控器（封装独立进程）"""
    
    def __init__(self, poll_interval=0.01, hold_ms=500):
        self.poll_interval = poll_interval
        self.hold_seconds = hold_ms / 1000.0        # ★ 新增：保持时间（秒）
        self.shared_value = Value('i', 1)           # GPIO电平
        self.shared_fd = Value('i', -1)             # fd 编号
        self.shared_press_time = Value('d', 0.0)    # ★ 新增：按下时间戳
        self.process = None
        
        self.last_consumed_time = 0.0
        self.debounce_seconds = 1.0                 # 1秒防抖冷却时间
    
    def start(self):
        """启动GPIO监控进程"""
        self.process = Process(
            target=gpio_monitor_process,
            args=(self.shared_value, self.shared_fd, self.shared_press_time, self.poll_interval),
            daemon=True
        )
        self.process.start()
        
        while self.shared_fd.value < 0:
            time.sleep(0.001)
        
        print(f"[GPIO] Button monitor started (GPIO{GPIO_NUM}, fd={self.shared_fd.value}, poll={self.poll_interval}s, hold={self.hold_seconds*1000:.0f}ms)")
    def get_state(self) -> int:
        """
        获取按钮状态
        
        Returns:
            1: 按钮松开
            0: 按钮按下
        """
        return self.shared_value.value
    
    def is_pressed(self) -> bool:
        """
        检测按钮是否按下，并保持 hold_ms 时间内的按下状态。
        
        - 按下瞬间：shared_value=0 → 返回 True
        - 500ms 内：仍返回 True（即使按钮已松开）
        - 超过 500ms：自动返回 False
        
        Returns:
            True:  按钮按下仍在保持窗口内
            False: 按钮未按下，或保持时间已过
        """
        # 物理状态必须是按下
        if self.shared_value.value != 0:
            return False
        
        # 检查是否在保持窗口内
        press_time = self.shared_press_time.value
        if press_time <= 0.0:
            return False
        
        elapsed = time.time() - press_time
        return elapsed < self.hold_seconds
    
    # 新增方法，放置在 is_pressed() 之后
    def consume_press(self) -> bool:
        """
        边沿检测：按钮按下时返回 True，并清除按下状态（一次性事件）。
        加入 1.0 秒防抖，防止单次物理按压触发多次事件。
        """
        now = time.time()
        if now - self.last_consumed_time < self.debounce_seconds:
            # 冷却期内如果有弹跳（bounce），直接吃掉信号，不触发有效事件
            with self.shared_press_time.get_lock():
                self.shared_press_time.value = 0.0
            return False

        if self.shared_value.value != 0:
            return False
        press_time = self.shared_press_time.value
        if press_time <= 0.0:
            return False
        elapsed = now - press_time
        if elapsed >= self.hold_seconds:
            return False
        
        # 消费：清除 press_time，并记录本次消费时间
        with self.shared_press_time.get_lock():
            self.shared_press_time.value = 0.0
        self.last_consumed_time = now
        return True

    def wait_for_press(self, timeout=None) -> bool:
        """
        等待一次按钮按下事件（阻塞）。
        
        Args:
            timeout: 超时时间(秒)，None 表示无限等待。
            
        Returns:
            True:  检测到按下
            False: 超时未检测到
        """
        fd = self.shared_fd.value
        if fd < 0:
            return False

        p = select.poll()
        p.register(fd, select.POLLPRI)

        # 先记录当前状态，避免重复触发
        start_state = self.shared_value.value

        if timeout is not None:
            deadline = time.time() + timeout
            remaining = timeout * 1000  # poll 单位是毫秒
            while True:
                result = p.poll(int(remaining))
                if result:
                    # 电平变化事件触发
                    os.lseek(fd, 0, os.SEEK_SET)
                    state = int(os.read(fd, 1))
                    os.lseek(fd, 0, os.SEEK_SET)
                    if state == 0:  # 按下（低电平）
                        return True
                    # 如果是松开，继续等待
                else:
                    return False
                remaining = (deadline - time.time()) * 1000
                if remaining <= 0:
                    return False
        else:
            while True:
                result = p.poll(-1)  # 永久阻塞直到事件
                os.lseek(fd, 0, os.SEEK_SET)
                state = int(os.read(fd, 1))
                os.lseek(fd, 0, os.SEEK_SET)
                if state == 0:
                    return True
    
    def stop(self):
        """停止GPIO监控进程"""
        if self.process and self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=1)
            print("[GPIO] Button monitor stopped")
        
        # 清理GPIO
        try:
            with open("/sys/class/gpio/unexport", "w") as f:
                f.write(str(GPIO_NUM))
        except:
            pass



try:
    import wiringpi
    from wiringpi import GPIO
    WIRINGPI_AVAILABLE = True
except ImportError:
    import logging
    logging.warning("wiringpi module not found. RGB LED will not function.")
    WIRINGPI_AVAILABLE = False
    GPIO = None

class HandcapRGB:
    def __init__(self, pin_r=9, pin_g=10, pin_b=13):
        self.pin_r = pin_r
        self.pin_g = pin_g
        self.pin_b = pin_b
        self.cmd_queue = queue.Queue(maxsize=10)
        
        if WIRINGPI_AVAILABLE:
            wiringpi.wiringPiSetup()
            wiringpi.pinMode(self.pin_r, GPIO.OUTPUT)
            wiringpi.pinMode(self.pin_g, GPIO.OUTPUT)
            wiringpi.pinMode(self.pin_b, GPIO.OUTPUT)
            self._set_hardware(0, 0, 0)
            
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _set_hardware(self, r, g, b):
        try:
            wiringpi.digitalWrite(self.pin_r, GPIO.HIGH if r else GPIO.LOW)
            wiringpi.digitalWrite(self.pin_g, GPIO.HIGH if g else GPIO.LOW)
            wiringpi.digitalWrite(self.pin_b, GPIO.HIGH if b else GPIO.LOW)
        except Exception:
            pass

    def _worker(self):
        while self.running:
            try:
                r, g, b = self.cmd_queue.get(timeout=0.5)
                if WIRINGPI_AVAILABLE:
                    self._set_hardware(r, g, b)
            except queue.Empty:
                pass
            except Exception:
                pass

    def turn_off_all(self):
        self._enqueue(0, 0, 0)

    def set_color(self, r, g, b):
        self._enqueue(r, g, b)

    def _enqueue(self, r, g, b):
        try:
            # 清空旧命令，只保留最新状态
            while not self.cmd_queue.empty():
                try:
                    self.cmd_queue.get_nowait()
                except queue.Empty:
                    break
            self.cmd_queue.put_nowait((r, g, b))
        except queue.Full:
            pass

    def set_recording(self):
        self.set_color(0, 1, 0)
        
    def set_paused(self):
        self.set_color(0, 0, 1)
        
    def set_error(self):
        self.set_color(1, 0, 0)
        
    def cleanup(self):
        self.running = False
        if WIRINGPI_AVAILABLE:
            self._set_hardware(0, 0, 0)
            try:
                wiringpi.pinMode(self.pin_r, GPIO.INPUT)
                wiringpi.pinMode(self.pin_g, GPIO.INPUT)
                wiringpi.pinMode(self.pin_b, GPIO.INPUT)
            except:
                pass


class HandCapEnv:
    def __init__(self, tactile_camera_dev_path_dict, wrist_camera_dev_path_dict, force_sensor_dev_path, save_dir, shm_manager, fps=24, *,
                 resolution=(768, 768), tactile_resolution=(224, 224), tactile_capture_resolution=(640, 480), buffer_size=1300):
        self.resolution = resolution
        self.tactile_resolution = tuple(tactile_resolution)
        self.tactile_capture_resolution = tuple(tactile_capture_resolution)
        self.fps = fps
        
        self.keyboard_events = []
        self.keyboard_lock = threading.Lock()
        self.keyboard_thread = None
        self.stop_keyboard_monitoring = False
        
        self.episode_id = 0
        self.episode_dir = ""  # 当前 episode 保存路径
        
        self.stop_event = threading.Event()
        
        # ── 缓存上一次有效帧，防止 peek() 为空时输出黑图 ──
        self.last_tactile_frames = {}
        self.last_wrist_frame = np.zeros((224, 224, 3), dtype=np.uint8)

        print("initializing Angle Sensor...")
        self.angle_sensor = AngleSensor(
            shm_manager=shm_manager,
            capture_fps=fps,
            put_fps=fps,
            get_max_k=buffer_size,
            verbose=True
        )
        
        camera_dict = {}
        for camera_name, (camera_cls, v4l_path) in tactile_camera_dev_path_dict.items():
            camera_dict[camera_name] = TactileCamera(
                camera_cls=camera_cls,
                dev_video_path=v4l_path,
                shm_manager=shm_manager,
                resolution=self.tactile_resolution,
                capture_resolution=self.tactile_capture_resolution,
                capture_fps=fps,
                put_fps=fps,
                get_max_k=buffer_size,
                cap_buffer_size=1,
                # put_downsample=False,
                verbose=True
            )
        
        wrist_camera_dict = {}
        print("initializing Wrist Camera...")
        for camera_name, (camera_cls, camera_params_dict) in wrist_camera_dev_path_dict.items():
            wrist_camera_dict[camera_name] = MVSWristCamera(
                camera_cls=camera_cls,
                cam_dict=camera_params_dict,
                shm_manager=shm_manager,
                resolution=resolution,
                capture_fps=fps,
                put_fps=fps,
                get_max_k=buffer_size,
                cap_buffer_size=1,
                # put_downsample=False,
                verbose=True
            )
            
        force_dict = {}
        print("initializing Force Sensor...")
        for force_name, (force_cls, smbus_id, i2c_address) in force_sensor_dev_path.items():
            force_dict[force_name] = ForceSensor(
                force_cls = force_cls,
                smbus_id=smbus_id,
                i2c_address=i2c_address,
                shm_manager=shm_manager,
                capture_fps=fps,
                put_fps=fps,
                get_max_k=buffer_size,
                verbose=True
            )
            
            
        self.camera_dict = camera_dict
        
        self.wrist_camera_dict = wrist_camera_dict
        self.force_dict = force_dict
        self.last_camera_data = {k: None for k in camera_dict.keys()}
        self.last_force_data = {k: None for k in force_dict.keys()}
        self.last_wrist_camera_data = {k: None for k in wrist_camera_dict.keys()}
        self.health_state = {'last_wc': {}, 'last_time': {}}
        self.save_dir = save_dir
        self.global_step = 0
    
    # ======== start-stop API =============
    @property
    def is_ready(self):
        ready_flag = True
        for camera in self.camera_dict.values():
            ready_flag = ready_flag and camera.is_ready
        
        for wrist_camera in self.wrist_camera_dict.values():
            ready_flag = ready_flag and wrist_camera.is_ready

        for force in self.force_dict.values():
            ready_flag = ready_flag and force.is_ready
        
        ready_flag = ready_flag and self.angle_sensor.is_ready
        return ready_flag
    
    def start(self, wait=True):
        
        self.angle_sensor.start(wait=False)
        time.sleep(1.0) # Add delay
        for camera in self.camera_dict.values():
            camera.start(wait=False)
            time.sleep(0.5)
        for wrist_camera in self.wrist_camera_dict.values():
            wrist_camera.start(wait=False)
            time.sleep(0.5)
        
        for force in self.force_dict.values():
            force.start(wait=False)
        if wait:
            self.start_wait()

    def stop(self, wait=True):
        for camera in self.camera_dict.values():
            camera.stop(wait=False)
        for wrist_camera in self.wrist_camera_dict.values():
            wrist_camera.stop(wait=False)
        for force in self.force_dict.values():
            force.stop(wait=False)
        self.angle_sensor.stop(wait=False)
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.angle_sensor.start_wait()
        for camera in self.camera_dict.values():
            camera.start_wait()
        for wrist_camera in self.wrist_camera_dict.values():
            wrist_camera.start_wait()
        for force in self.force_dict.values():
            force.start_wait()
        
    def stop_wait(self):
        for camera in self.camera_dict.values():
            camera.end_wait()
        for wrist_camera in self.wrist_camera_dict.values():
            wrist_camera.end_wait()
        for force in self.force_dict.values():
            force.end_wait()
        self.angle_sensor.end_wait()

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= async env API ===========
    def get_and_save_data(self):
        """
        【数据保存专用接口 — 优先级最高】
        - 使用 get_all() 阻塞等待直到有数据
        - 任意一个 sensor 数据失败 → 跳过整个帧，不保存不完整的数据
        - 保存操作在独立子线程中执行，不阻塞后续数据获取
        - UDP 线程完全独立，使用 peek() 不受影响
        """
        assert self.is_ready

        # ── 第一步：消费所有传感器数据（阻塞等待） ──────────────────────
        all_ok = True
        for camera_name, camera in self.camera_dict.items():
            try:
                self.last_camera_data[camera_name] = camera.get_all()
            except queue.Empty:
                print(f"\033[91m[SAVE ERROR]\033[0m Tactile camera '{camera_name}': no data in buffer, skipping frame.")
                all_ok = False
            except Exception as e:
                print(f"\033[91m[SAVE ERROR]\033[0m Tactile camera '{camera_name}': {e}, skipping frame.")
                all_ok = False
                self.last_camera_data[camera_name] = None

        for camera_name, wrist_camera in self.wrist_camera_dict.items():
            try:
                self.last_wrist_camera_data[camera_name] = wrist_camera.get_all()
            except queue.Empty:
                print(f"\033[91m[SAVE ERROR]\033[0m Wrist camera '{camera_name}': no data in buffer, skipping frame.")
                all_ok = False
            except Exception as e:
                print(f"\033[91m[SAVE ERROR]\033[0m Wrist camera '{camera_name}': {e}, skipping frame.")
                all_ok = False
                self.last_wrist_camera_data[camera_name] = None

        for force_name, force in self.force_dict.items():
            try:
                self.last_force_data[force_name] = force.get_all()
            except queue.Empty:
                print(f"\033[91m[SAVE ERROR]\033[0m Force sensor '{force_name}': no data in buffer, skipping frame.")
                all_ok = False
            except Exception as e:
                print(f"\033[91m[SAVE ERROR]\033[0m Force sensor '{force_name}': {e}, skipping frame.")
                all_ok = False
                self.last_force_data[force_name] = None

        try:
            self.last_angle_data = self.angle_sensor.get_all()
            if self.last_angle_data is None:
                print(f"\033[91m[SAVE ERROR]\033[0m Angle sensor: returned None, skipping frame.")
                all_ok = False
        except queue.Empty:
            print(f"\033[91m[SAVE ERROR]\033[0m Angle sensor: no data in buffer, skipping frame.")
            all_ok = False
            self.last_angle_data = None
        except Exception as e:
            print(f"\033[91m[SAVE ERROR]\033[0m Angle sensor: {e}, skipping frame.")
            all_ok = False
            self.last_angle_data = None

        # 任意 sensor 数据失败 → 跳过整个帧，不保存不完整数据
        if not all_ok:
            self.global_step += 1
            return

        # ── 第二步：异步保存（不阻塞数据获取） ─────────────────────────
        print("Saving data...")
        threads = []

        for camera_name, camera_data in self.last_camera_data.items():
            t = threading.Thread(target=self._save_tactile_video_and_json,
                                args=("tactile/" + camera_name + "tactile", camera_data))
            threads.append(t)
            t.start()

        for camera_name, camera_data in self.last_wrist_camera_data.items():
            t = threading.Thread(target=self._save_video_and_json,
                                args=("wrist/" + camera_name + "camera", camera_data))
            threads.append(t)
            t.start()

        for force_name, force_data in self.last_force_data.items():
            t = threading.Thread(target=self._save_force_data,
                                args=("force/" + force_name + "force", force_data))
            threads.append(t)
            t.start()

        if self.last_angle_data:
            t = threading.Thread(target=self._save_angle_data, args=(self.last_angle_data,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        self.global_step += 1
        self.last_camera_data = {k: None for k in self.camera_dict.keys()}
        self.last_wrist_camera_data = {k: None for k in self.wrist_camera_dict.keys()}
        self.last_force_data = {k: None for k in self.force_dict.keys()}
        self.last_angle_data = None
    
    def _save_video_and_json(self, camera_name, camera_data):
        start_time = time.monotonic()
        if camera_data is None:
            print(f"[WARN] _save_video_and_json: {camera_name} data is None, skipping.")
            return
        base_dir = self.episode_dir if self.episode_dir else self.save_dir
        save_dir = os.path.join(base_dir, camera_name.split('/')[0])
        os.makedirs(save_dir, exist_ok=True)
        video_writer = cv2.VideoWriter(
            os.path.join(save_dir, f"{camera_name.split('/')[1]}_{self.global_step}.mp4"),
            cv2.VideoWriter_fourcc(*'mp4v'),
            self.fps, self.resolution)

        video_shape = camera_data['color'].shape

        assert len(video_shape) == 4, str(video_shape)  # T,w,h,c

        for frame in camera_data['color']:
            video_writer.write(frame)
            time.sleep(0.002)  # Yield CPU to UDP threads to prevent network stutter
        video_writer.release()
        with open(os.path.join(save_dir, f"{camera_name.split('/')[1]}_{self.global_step}.json"), 'w') as json_file:
            timestamps = camera_data["timestamp"].tolist()
            json.dump(timestamps, json_file)

        print("Saved camera data")
    
    def _save_tactile_video_and_json(self, camera_name, camera_data):
        start_time = time.monotonic()
        if camera_data is None:
            print(f"[WARN] _save_tactile_video_and_json: {camera_name} data is None, skipping.")
            return
        base_dir = self.episode_dir if self.episode_dir else self.save_dir
        save_dir = os.path.join(base_dir, camera_name.split('/')[0])
        os.makedirs(save_dir, exist_ok=True)
        video_shape = camera_data['color'].shape

        assert len(video_shape) == 4, str(video_shape)  # T,w,h,c
        tactile_video_size = (video_shape[2], video_shape[1])
        video_writer = cv2.VideoWriter(
            os.path.join(save_dir, f"{camera_name.split('/')[1]}_{self.global_step}.mp4"),
            cv2.VideoWriter_fourcc(*'mp4v'),
            self.fps, tactile_video_size)

        for frame in camera_data['color']:
            video_writer.write(frame)
            time.sleep(0.002)  # Yield CPU to UDP threads to prevent network stutter
        video_writer.release()
        with open(os.path.join(save_dir, f"{camera_name.split('/')[1]}_{self.global_step}.json"), 'w') as json_file:
            timestamps = camera_data["timestamp"].tolist()
            json.dump(timestamps, json_file)

        print("Saved camera data")

    def _save_angle_data(self, angle_data):
        if angle_data is None:
            print(f"[WARN] _save_angle_data: data is None, skipping.")
            return
        start_time = time.monotonic()
        base_dir = self.episode_dir if self.episode_dir else self.save_dir
        save_dir = os.path.join(base_dir, "angle")
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, f"angle_{self.global_step}.json"), 'w') as json_file:
            angles = angle_data['angle'].tolist()
            data = angle_data['data'].tolist()
            timestamps = angle_data["timestamp"].tolist()
            json.dump({'angles': angles, 'data':data, 'timestamps': timestamps}, json_file)
        print("\033[92m Successfully saved angle&pose data \033[0m")

    def _save_force_data(self, force_name, force_data):
        if force_data is None:
            print(f"[WARN] _save_force_data: {force_name} data is None, skipping.")
            return
        start_time = time.monotonic()
        base_dir = self.episode_dir if self.episode_dir else self.save_dir
        save_dir = os.path.join(base_dir, force_name.split('/')[0])
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, f"{force_name.split('/')[1]}_{self.global_step}.json"), 'w') as json_file:
            forces = force_data['force'].tolist()
            timestamps = force_data["timestamp"].tolist()
            json.dump({'forces': forces, 'timestamps': timestamps}, json_file)
        print("\033[92m Successfully saved force data \033[0m")
    
        # ========= UDP 实时帧接口（优先级次之） ==========
    def get_current_frame(self):
        """
        获取所有 sensor 的最新帧，用于 UDP 传输。
        - 使用 peek() 非破坏性读取，永远不消费 buffer
        - 若 buffer 暂无数据，返回零图 / 空字典，不阻塞
        - 保存线程使用 get_all() 消费数据，与本接口完全独立
        """
        result = {
            'force': {},
            'angle': None,
            'sys_time': time.time(),
        }
        current_time = result['sys_time']
        timeout_sec = 1.0

        for camera_name, camera in self.camera_dict.items():
            try:
                # 获取队列最新帧（解决 peek() 总是获取列首旧帧导致静止长达10秒的问题）
                rb = camera.ring_buffer
                wc = rb.write_counter.load()
                rc = rb.read_counter.load()
                if wc > rc:
                    latest_idx = (wc - 1) % rb.buffer_size
                    latest_color = rb.shared_arrays['color'].get()[latest_idx].copy()
                    self.last_tactile_frames[camera_name] = {'color': latest_color}
            except Exception:
                pass

        # wrist cameras — 只读最新帧，非破坏性
        for camera_name, wrist_camera in self.wrist_camera_dict.items():
            try:
                rb = wrist_camera.ring_buffer
                wc = rb.write_counter.load()
                rc = rb.read_counter.load()
                if wc > rc:
                    latest_idx = (wc - 1) % rb.buffer_size
                    raw = rb.shared_arrays['color'].get()[latest_idx].copy()
                    self.last_wrist_frame = cv2.resize(raw, (224, 224), interpolation=cv2.INTER_AREA)
            except Exception:
                pass

        # force sensors — 状态检测：数据是否更新
        for force_name, force in self.force_dict.items():
            status = False
            try:
                wc = force.ring_buffer.write_counter.load()
                last_wc = self.health_state['last_wc'].get(force_name, -1)
                last_time = self.health_state['last_time'].get(force_name, current_time)
                
                if wc > last_wc:
                    self.health_state['last_wc'][force_name] = wc
                    self.health_state['last_time'][force_name] = current_time
                    status = True
                else:
                    if current_time - last_time < timeout_sec:
                        status = True
            except Exception:
                pass
            result['force'][force_name] = {'status': status and force.is_ready}

        # angle sensor — 状态检测：数据是否更新
        angle_status = False
        try:
            wc = self.angle_sensor.ring_buffer.write_counter.load()
            last_wc = self.health_state['last_wc'].get('angle', -1)
            last_time = self.health_state['last_time'].get('angle', current_time)
            
            if wc > last_wc:
                self.health_state['last_wc']['angle'] = wc
                self.health_state['last_time']['angle'] = current_time
                angle_status = True
            else:
                if current_time - last_time < timeout_sec:
                    angle_status = True
        except Exception:
            pass
        result['angle'] = {'status': angle_status and self.angle_sensor.is_ready}

        return self.last_wrist_frame, self.last_tactile_frames, result

def check_time_sync(server_ip, duration=10, max_offset_ms=10.0):
    print(f"\n[Time Sync] 开始执行 {duration} 秒的时间同步检查 (目标: {server_ip})...")
    print(f"[Time Sync] 标准: 误差需小于 {max_offset_ms}ms")
    
    ntp_delta = 2208988800
    
    def get_single_ntp_sample(host, port=123):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(0.5)
        msg = b'\x1b' + 47 * b'\0'
        try:
            t1 = time.time()
            sock.sendto(msg, (host, port))
            data, addr = sock.recvfrom(1024)
            t4 = time.time()
            if data:
                unpacked = struct.unpack('!12I', data)
                t2 = unpacked[8] - ntp_delta + unpacked[9] / 2**32
                t3 = unpacked[10] - ntp_delta + unpacked[11] / 2**32
                offset = ((t2 - t1) + (t3 - t4)) / 2
                delay = (t4 - t1) - (t3 - t2)
                return offset, delay
        except:
            return None
        finally:
            sock.close()

    def get_ntp_offset(host, samples=NTP_CHECK_SAMPLES):
        valid_samples = []
        for sample_idx in range(samples):
            sample = get_single_ntp_sample(host)
            if sample is not None:
                valid_samples.append(sample)
            if sample_idx < samples - 1:
                time.sleep(NTP_CHECK_SAMPLE_GAP_SEC)

        if not valid_samples:
            return None

        # The lowest-delay sample is least affected by one-off network/OS scheduling jitter.
        return min(valid_samples, key=lambda item: item[1])
            
    start_time = time.time()
    exceed_count = 0
    total_checks = 0
    
    print("-" * 78)
    print(f"{'时间':<10} | {'偏移量 (ms)':<15} | {'RTT (ms)':<12} | {'状态':<10} ")
    print("-" * 78)
    
    while time.time() - start_time < duration:
        sample = get_ntp_offset(server_ip)
        current_time_str = time.strftime("%H:%M:%S", time.localtime())
        
        if sample is not None:
            offset, delay = sample
            offset_ms = offset * 1000
            delay_ms = delay * 1000
            abs_ms = abs(offset_ms)
            status = "✅ 正常" if abs_ms <= max_offset_ms else "❌ 异常"
            color_code = "\033[92m" if abs_ms <= max_offset_ms else "\033[91m"
            reset_code = "\033[0m"
            
            print(
                f"{current_time_str:<10} | {color_code}{offset_ms:10.4f} ms{reset_code}    | "
                f"{delay_ms:9.3f} ms | {status:<10} "
            )
            if abs_ms > max_offset_ms:
                exceed_count += 1
        else:
            print(f"{current_time_str:<10} | {'TIMEOUT':<15} | {'-':<12} | ⚠️ 超时      ")
            exceed_count += 1
            
        total_checks += 1
        time.sleep(1)
        
    if exceed_count > 0:
        print(f"\n\033[91m[警告] 在过去 {duration} 秒的测试中，有 {exceed_count}/{total_checks} 次时间同步误差超过 {max_offset_ms}ms (或超时)！\033[0m")
        print("\033[91m请检查网络或时间同步服务 (Windows Time / chrony)！\033[0m")
        input("按 Enter 键忽略警告并继续，或按 Ctrl+C 退出...")
    else:
        print("\033[92m[Time Sync] 时间同步检查通过！\033[0m\n")


def main(args):
    script_start_time = time.time()
    total_effective_time = 0.0
    current_episode_start = None
    # ── 执行时间同步检查 ──────────────────────────────────────────────────
    if NTP_HOST:
        check_time_sync(server_ip=NTP_HOST, duration=10, max_offset_ms=10.0)
    else:
        print("[Time Sync] Set HANDCAP_NTP_HOST to check synchronization with the tracker host.")

    # ── 启动GPIO按钮监控进程 ─────────────────────────────────────────────
    gpio_monitor = GPIOButtonMonitor(poll_interval=0.01)
    gpio_monitor.start()
    
    # ── 初始化 RGB 灯 ────────────────────────────────────────────────────
    rgb_indicator = HandcapRGB()
    # 启动时默认为关闭状态（不开启灯光）
    
    # 1. UDP transfer setup
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # Allow fast restart after abnormal exit (avoid Errno 98 on bind).
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    except (AttributeError, OSError):
        # Some kernels/platforms do not support SO_REUSEPORT.
        pass
    try:
        sock.bind((UDP_IP, UDP_PORT))
    except OSError as e:
        print(f"❌ Failed to bind UDP {UDP_IP}:{UDP_PORT}: {e}")
        print("提示: 可能已有旧进程占用端口，请先结束旧进程后重试。")
        raise
    configure_udp_socket(sock)
    print(f"📹 UDP Server listening on {UDP_IP}:{UDP_PORT}")
    
    # 双向握手
    sock.settimeout(30.0)
    print("⏳ Waiting for PC Client handshake (HELLO)...")
    handshake_ok = False
    client_clock_offset = 0.0  # client_time - this_process_time; applied only to sys_time_aligned
    clock_sync_applied = False
    try:
        while not handshake_ok:
            data, client_addr = sock.recvfrom(65535)
            print(f"Received from {client_addr}: {data[:50] if len(data) > 50 else data}")
            hello_msg = None
            if data != b"HELLO":
                try:
                    hello_msg = json.loads(data.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    hello_msg = None

            if data == b"HELLO" or (isinstance(hello_msg, dict) and hello_msg.get("type") == "HELLO_SYNC"):
                server_rx_time = time.time()
                print(f"🚀 Client connected from {client_addr}. Sending HELLO_REPLY...")
                if isinstance(hello_msg, dict) and hello_msg.get("type") == "HELLO_SYNC":
                    reply_msg = {
                        "type": "HELLO_REPLY_SYNC",
                        "client_time": float(hello_msg.get("client_time", 0.0)),
                        "server_rx_time": server_rx_time,
                        "server_tx_time": time.time(),
                    }
                    sock.sendto(json.dumps(reply_msg, separators=(",", ":")).encode("utf-8"), client_addr)
                else:
                    sock.sendto(b"HELLO_REPLY", client_addr)
                
                # 等待ACK确认
                sock.settimeout(5.0)
                try:
                    ack_data, _ = sock.recvfrom(1024)
                    ack_msg = None
                    if ack_data != b"ACK":
                        try:
                            ack_msg = json.loads(ack_data.decode("utf-8"))
                        except (UnicodeDecodeError, json.JSONDecodeError):
                            ack_msg = None

                    if isinstance(ack_msg, dict) and ack_msg.get("type") == "ACK_SYNC":
                        client_clock_offset = float(ack_msg.get("client_minus_server_offset", 0.0))
                        clock_sync_applied = True
                        print(
                            "✅ Handshake complete with clock sync! "
                            f"client-server offset={client_clock_offset:.6f}s"
                        )
                        handshake_ok = True
                    elif ack_data == b"ACK":
                        print("✅ Handshake complete! Starting data stream...")
                        handshake_ok = True
                    else:
                        print(f"⚠️ Unexpected message: {ack_data}")
                except socket.timeout:
                    print("⚠️ ACK timeout, continuing anyway...")
                    handshake_ok = True
            else:
                print(f"⚠️ Ignoring non-HELLO message (len={len(data)}), waiting for HELLO...")
    except socket.timeout:
        print("❌ Handshake timeout! No client connected.")
        sock.close()
        exit(1)
    sock.settimeout(None)


    # define tactile device
    tactile_camera_dev_path_dict = {}
    for side, device in TACTILE_CAMERA.items():
        tactile_camera_dev_path_dict[side] = [TactileUSBCamera, device]
    
    # define force device
    force_dev_path_dict = {}
    for side, device in FORCE_SENSOR.items():
        force_dev_path_dict[side] = [ForceReader, device['smbus_id'], device['i2c_address']]
    
    # define wrist camera device
    wrist_camera_dev_path_dict = {}
    for side, device in WRIST_CAMERA.items():
        wrist_camera_dev_path_dict[side] = [MVSCamera, device]
    time.sleep(1.0) # warmup camera

    save_dir = "data/handcap_"  + str(args.cap_id)
    assert not os.path.exists(save_dir), "path exists"
    os.makedirs(save_dir)

    # 按钮控制状态变量
    is_recording = False
    episode_index = 0
    recording_lock = threading.Lock()
    
    save_per_second = 1.0 / args.fps  # interval duration in seconds (e.g., 0.05s for 20Hz)
    
    frame_id = 0

    with SharedMemoryManager() as shm_manager:
        with HandCapEnv(
            tactile_camera_dev_path_dict, wrist_camera_dev_path_dict,
            force_dev_path_dict,
            save_dir,
            shm_manager,
            fps=args.fps,
            tactile_resolution=(args.tactile_width, args.tactile_height),
            tactile_capture_resolution=(args.tactile_capture_width, args.tactile_capture_height),
        ) as env:

            # ── Save thread: 10s chunked save, based on is_recording state ───────
            def save_loop(chunk_interval_sec):
                try:
                    import os
                    os.setpriority(os.PRIO_PROCESS, os.gettid(), 10)
                except Exception:
                    pass
                last_save_time = time.monotonic()
                was_recording = False
                while not env.stop_event.is_set():
                    time.sleep(0.02)  # Check state frequently
                    
                    with recording_lock:
                        curr_recording = is_recording
                    
                    now = time.monotonic()
                    time_to_save = (now - last_save_time >= chunk_interval_sec)
                    force_save = (was_recording and not curr_recording)
                    
                    if curr_recording and time_to_save:
                        print(f'[DEBUG] Chunk Time! Saving Episode {episode_index} chunk {env.global_step}')
                        env.get_and_save_data()
                        last_save_time = now
                        
                    elif force_save:
                        print(f'[DEBUG] Recording Stopped! Flushing final frames for Episode {episode_index} chunk {env.global_step}')
                        env.get_and_save_data()
                        last_save_time = now
                        
                    elif not curr_recording:
                        # Clear buffers quietly to prevent memory explosion
                        for camera in env.camera_dict.values():
                            try: camera.get_all()
                            except Exception: pass
                        for wrist_camera in env.wrist_camera_dict.values():
                            try: wrist_camera.get_all()
                            except Exception: pass
                        for force in env.force_dict.values():
                            try: force.get_all()
                            except Exception: pass
                        try: env.angle_sensor.get_all()
                        except Exception: pass
                        last_save_time = now  # Reset timer so it doesn't instantly chunk on next cycle start
                        
                    was_recording = curr_recording

                if was_recording:
                    print(f'[DEBUG] Final flush on exit! Saving remaining frames...')
                    env.get_and_save_data()

            save_thread = threading.Thread(target=save_loop, args=(10.0,), daemon=True)
            save_thread.start()

            # ── Async JPEG encoding thread ─────────────────────────────────────────
            JPEG_QUALITY = 75  # 提高传输质量
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]

            # UDP preview queues only. They drop stale frames under load and never touch save buffers.
            encode_queue = queue.Queue(maxsize=ASYNC_QUEUE_SIZE)
            send_queue = queue.Queue(maxsize=UDP_SEND_QUEUE_SIZE)
            send_stop = threading.Event()

            _dbg_send_queue_full_count_ref = [0]  # 编码线程和发送线程共享计数

            def encode_worker():
                """Dedicated thread for JPEG encoding - runs at full speed without blocking main loop"""
                _encode_count = 0
                _last_encode_report = time.time()
                _encode_times = []  # 每次编码耗时
                _dbg_encode_queue_wait = []  # 帧在队列中的等待时间
                _encode_submit_times = {}  # frame_id -> 入队时间
                _encode_first_report = True
                while not env.stop_event.is_set():
                    try:
                        # Non-blocking get with timeout
                        frame_data = encode_queue.get(timeout=0.1)
                        if frame_data is None:
                            continue

                        if len(frame_data) == 6:
                            wrist_frame, left_color, right_color, metadata_data, frame_id_local, submit_time = frame_data
                        else:
                            wrist_frame, left_color, right_color, metadata_data, frame_id_local = frame_data
                            submit_time = time.time()
                        queue_wait_ms = (time.time() - submit_time) * 1000
                        _dbg_encode_queue_wait.append(queue_wait_ms)
                        encode_start = time.time()

                        # Resize for UDP transmission (112x112 to save bandwidth)
                        left_udp = cv2.resize(left_color, (112, 112), interpolation=cv2.INTER_AREA)
                        right_udp = cv2.resize(right_color, (112, 112), interpolation=cv2.INTER_AREA)

                        # JPEG encoding
                        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
                        _, left_jpg = cv2.imencode('.jpg', left_udp, encode_param)
                        _, right_jpg = cv2.imencode('.jpg', right_udp, encode_param)
                        _, wrist_jpg = cv2.imencode('.jpg', wrist_frame, encode_param)

                        encode_time = time.time() - encode_start
                        total_encode_ms = (time.time() - submit_time) * 1000
                        _encode_times.append(encode_time)
                        _encode_count += 1

                        if _DEBUG_ENCODE_LOG and (_encode_count <= 30 or _encode_count % 100 == 0):
                            print(f"[Encode] fid={frame_id_local} | "
                                  f"queue_wait={queue_wait_ms:.1f}ms | "
                                  f"encode={encode_time*1000:.1f}ms | "
                                  f"total={total_encode_ms:.1f}ms | "
                                  f"wrist={len(wrist_jpg)}B left={len(left_jpg)}B right={len(right_jpg)}B")

                        # Build payload
                        metadata_data['wrist_jpeg_len'] = len(wrist_jpg)
                        metadata_data['tactile_left_len'] = len(left_jpg)
                        metadata_data['tactile_right_len'] = len(right_jpg)
                        # sys_time already set at capture time in main loop
                        metadata_data['wrist_size'] = len(wrist_jpg)
                        metadata_data['tactile_size'] = len(left_jpg) + len(right_jpg)
                        metadata_data['encode_time'] = encode_time

                        raw_payload = wrist_jpg.tobytes() + left_jpg.tobytes() + right_jpg.tobytes()
                        metadata_bytes = json.dumps(metadata_data).encode('utf-8')
                        len_bytes = len(metadata_bytes).to_bytes(4, byteorder='big')
                        full_payload = len_bytes + metadata_bytes + raw_payload
                        total_payload_len = len(full_payload)
                        num_chunks = math.ceil(total_payload_len / MAX_PACKET_SIZE)

                        if _DEBUG_ENCODE_LOG and (_encode_count <= 30 or _encode_count % 100 == 0):
                            meta_len_bytes = len(metadata_bytes)
                            print(f"  └─ [Encode] meta={meta_len_bytes}B raw={len(raw_payload)}B total={total_payload_len}B "
                                  f"chunks={num_chunks} chunk_size={MAX_PACKET_SIZE}B")

                        # ── 发送给发送队列（保证最新帧策略，踢掉旧数据缩水延迟） ────────────────────────
                        try:
                            send_item = (frame_id_local, full_payload, num_chunks, total_payload_len)
                            while True:
                                try:
                                    send_queue.put_nowait(send_item)
                                    break
                                except queue.Full:
                                    try: 
                                        send_queue.get_nowait()
                                        _dbg_send_queue_full_count_ref[0] += 1
                                    except queue.Empty: pass
                        except Exception as e:
                            print(f"[Encode Worker] send_queue.put exception: {e}")

                        # Report encode stats every 5 seconds
                        now = time.time()
                        if now - _last_encode_report >= 5.0:
                            elapsed = now - _last_encode_report
                            encode_fps = _encode_count / elapsed if elapsed > 0 else 0
                            queue_avg = statistics.mean(_dbg_encode_queue_wait) if _dbg_encode_queue_wait else 0
                            queue_max = max(_dbg_encode_queue_wait) if _dbg_encode_queue_wait else 0
                            encode_avg_ms = statistics.mean(_encode_times) * 1000 if _encode_times else 0
                            encode_max_ms = max(_encode_times) * 1000 if _encode_times else 0
                            if _DEBUG_ENCODE_LOG or _encode_first_report:
                                print(f"\n=== [Encode 5s Report] ===")
                                print(f"  encode_fps={encode_fps:.1f}Hz | count={_encode_count} | elapsed={elapsed:.1f}s")
                                print(f"  encode_time: avg={encode_avg_ms:.1f}ms max={encode_max_ms:.1f}ms")
                                print(f"  queue_wait: avg={queue_avg:.1f}ms max={queue_max:.1f}ms")
                                print(f"  encode_queue_size={encode_queue.qsize()}/{ASYNC_QUEUE_SIZE}")
                                print(f"  send_queue_size={send_queue.qsize()}/{UDP_SEND_QUEUE_SIZE}")
                                print(f"  send_queue_full_total={_dbg_send_queue_full_count_ref[0]}")
                                print(f"==========================\n")
                                _encode_first_report = False
                            _encode_count = 0
                            _encode_times.clear()
                            _dbg_encode_queue_wait.clear()
                            _last_encode_report = now

                    except queue.Empty:
                        continue
                    except Exception as e:
                        print(f"[Encode Worker] Error: {e}")
                        import traceback
                        traceback.print_exc()
                        continue

            encode_thread = threading.Thread(target=encode_worker, daemon=True, name="JPEG-Encoder")
            encode_thread.start()

            # ── Async send thread ──────────────────────────────────────────────────
            def send_worker():
                _send_count = 0
                _send_packets = 0
                _send_bytes = 0
                _last_send_report = time.time()
                _send_first_report = True
                while not send_stop.is_set():
                    try:
                        frame_data = send_queue.get(timeout=0.05)
                        if frame_data is None:
                            continue
                        frame_id_local, full_payload, num_chunks, total_payload_len = frame_data
                        send_start = time.time()
                        total_len = len(full_payload)

                        if _DEBUG_SEND_LOG and (_send_count < 30 or _send_count % 100 == 0):
                            print(f"[Send] fid={frame_id_local} | payload={total_payload_len}B | chunks={num_chunks} "
                                  f"| ~pkt_size={total_payload_len/num_chunks:.0f}B")

                        for i in range(num_chunks):
                            start = i * MAX_PACKET_SIZE
                            chunk = full_payload[start:start + MAX_PACKET_SIZE]
                            # Use full 32-bit frame_id (no modulo wraparound!)
                            header = struct.pack("!IHH", frame_id_local, i, num_chunks)
                            pkt = header + chunk
                            sock.sendto(pkt, client_addr)
                            
                            # Yield after a short burst so saving/encoding threads keep CPU,
                            # but avoid fixed millisecond sleeps that create preview latency.
                            if i > 0 and i % UDP_SEND_BURST_CHUNKS == 0:
                                time.sleep(0)

                            _send_packets += 1
                            _send_bytes += len(pkt)
                            if _DEBUG_PACKET_LOG and _send_packets <= 50:
                                print(f"  └─ [SEND pkt#{_send_packets}] fid={frame_id_local} "
                                      f"chunk={i}/{num_chunks} size={len(pkt)}B")
                        _send_count += 1
                        send_time_ms = (time.time() - send_start) * 1000
                        if _DEBUG_SEND_LOG and (_send_count < 30 or _send_count % 100 == 0):
                            print(f"  └─ [Send Done] fid={frame_id_local} sent {num_chunks} chunks in {send_time_ms:.1f}ms")

                        # Report send stats every 5 seconds
                        now = time.time()
                        if now - _last_send_report >= 5.0:
                            elapsed = now - _last_send_report
                            send_fps = _send_count / elapsed if elapsed > 0 else 0
                            bw_mbps = (_send_bytes / (1024 * 1024)) / elapsed if elapsed > 0 else 0
                            if _DEBUG_SEND_LOG or _send_first_report:
                                print(f"\n=== [Send 5s Report] ===")
                                print(f"  send_fps={send_fps:.1f}Hz | frames_sent={_send_count}")
                                print(f"  packets_sent={_send_packets} | bytes_sent={_send_bytes} ({bw_mbps:.1f} MB/s)")
                                print(f"  avg_pkt_size={_send_bytes/max(_send_packets,1)}B | chunks_per_frame={_send_packets/max(_send_count,1):.1f}")
                                print(f"========================\n")
                                _send_first_report = False
                            _send_count = 0
                            _send_packets = 0
                            _send_bytes = 0
                            _last_send_report = now

                    except queue.Empty:
                        continue
                    except Exception as e:
                        print(f"[Send Worker] Error: {e}")
                        import traceback
                        traceback.print_exc()
                        continue

            send_thread = threading.Thread(target=send_worker, daemon=True)
            send_thread.start()

            # ── Main loop: data acquisition only, encoding is async ───────────────
            frame_id = 0
            _last_encode_time = time.time()
            _encode_interval = 1.0 / ENCODER_THREAD_FPS

            # ── 调试统计变量 ────────────────────────────────────────────────────
            _dbg_loop_count = 0
            _dbg_queue_full_count = 0     # 编码队列满的次数
            _dbg_send_queue_full_count = 0  # 发送队列满的次数
            _dbg_last_frame_print = -1
            _dbg_loop_times = []          # 主循环耗时记录
            _dbg_last_main_time = time.time()
            _dbg_encode_submit_count = 0  # 成功提交到编码队列的帧数
            _dbg_encode_skip_count = 0   # 跳过（队列满）的帧数
            _dbg_send_count = 0          # 实际发送的帧数
            _dbg_last_send_report = time.time()

            if _DEBUG_VERBOSE:
                print(f"[SENDER DEBUG] === Debug Configuration ===")
                print(f"  ENCODER_THREAD_FPS = {ENCODER_THREAD_FPS}")
                print(f"  _encode_interval   = {_encode_interval*1000:.1f}ms (即每 {_encode_interval:.3f}s 一帧)")
                print(f"  ASYNC_QUEUE_SIZE  = {ASYNC_QUEUE_SIZE}")
                print(f"  MAX_PACKET_SIZE   = {MAX_PACKET_SIZE}B")
                print(f"[SENDER DEBUG] ==============================\n")

            _light_has_started = False
            _current_light_state = "off"
            _sensor_error_since = None
            _sensor_ok_since = None
            _sensor_error_latched = False
            _last_sensor_error_report = 0.0
            _last_light_apply_time = 0.0

            try:
                while True:
                    loop_start = time.time()
                    _dbg_loop_count += 1
                    loop_delta_ms = (loop_start - _dbg_last_main_time) * 1000
                    _dbg_last_main_time = loop_start
                    _dbg_loop_times.append(loop_delta_ms)

                    if _DEBUG_TIMING_LOG and _dbg_loop_count <= 100:
                        print(f"[MainLoop#{_dbg_loop_count}] loop_delta={loop_delta_ms:.1f}ms | frame_id={frame_id}")

                    # Set timestamp at capture time (not after encoding)
                    capture_time = time.time()

                    # ── STAGE 1: Data sensing (camera + sensors) ─────────────────────
                    sense_start = time.time()
                    wrist_frame, tactile_frames_data, metadata_data = env.get_current_frame()
                    sense_time_ms = (time.time() - sense_start) * 1000

                    # Add timestamp at capture time. Keep raw sys_time for saved metadata,
                    # and provide receiver-clock aligned time for realtime latency display.
                    metadata_data['sys_time'] = capture_time
                    metadata_data['sys_time_aligned'] = capture_time + client_clock_offset
                    metadata_data['clock_sync_offset'] = client_clock_offset
                    metadata_data['clock_sync_applied'] = clock_sync_applied

                    if gpio_monitor.consume_press():
                        with recording_lock:
                            if not is_recording:
                                episode_index += 1
                                episode_dir = os.path.join(save_dir, f"episode_{episode_index:04d}")
                                
                                def _mkd(d):
                                    try: os.makedirs(d, exist_ok=True)
                                    except Exception: pass
                                threading.Thread(target=_mkd, args=(episode_dir,), daemon=True).start()
                                
                                is_recording = True
                                current_episode_start = time.time()
                                _light_has_started = True
                                _sensor_error_since = None
                                _sensor_ok_since = None
                                _sensor_error_latched = False
                                env.episode_dir = episode_dir
                                env.global_step = 0
                                print(f"\n=== Recording Episode {episode_index} ===")
                            else:
                                is_recording = False
                                if 'current_episode_start' in locals() and current_episode_start is not None:
                                    total_effective_time += time.time() - current_episode_start
                                    current_episode_start = None
                                _sensor_error_since = None
                                _sensor_ok_since = None
                                _sensor_error_latched = False
                                print(f"\n=== Episode {episode_index} Recording Done ===")

                        metadata_data['button_event'] = True
                    else:
                        metadata_data['button_event'] = False

                    with recording_lock:
                        metadata_data['collection_state'] = 1 if is_recording else 0
                        metadata_data['episode_index'] = episode_index

                    # ── STAGE 1.5: Sensor Status Check & Dynamic Light Control ─────────
                    # 根据是否产生按键逻辑以及传感器是否异常，动态判断灯的颜色并切换
                    sensor_error_sources = []
                    if metadata_data.get('angle', {}).get('status', False) == False:
                        sensor_error_sources.append("angle")
                    for force_name, force_status in metadata_data.get('force', {}).items():
                        if force_status.get('status', False) == False:
                            sensor_error_sources.append(f"force:{force_name}")

                    raw_sensor_error = bool(sensor_error_sources)
                    light_check_time = time.monotonic()
                    if raw_sensor_error:
                        _sensor_ok_since = None
                        if _sensor_error_since is None:
                            _sensor_error_since = light_check_time
                        elif (
                            not _sensor_error_latched
                            and light_check_time - _sensor_error_since >= RGB_SENSOR_ERROR_DEBOUNCE_SEC
                        ):
                            _sensor_error_latched = True
                            if light_check_time - _last_sensor_error_report >= 2.0:
                                print(
                                    "[RGB] Sensor error persisted; showing red. "
                                    f"sources={','.join(sensor_error_sources)}"
                                )
                                _last_sensor_error_report = light_check_time
                    else:
                        _sensor_error_since = None
                        if _sensor_ok_since is None:
                            _sensor_ok_since = light_check_time
                        elif (
                            _sensor_error_latched
                            and light_check_time - _sensor_ok_since >= RGB_SENSOR_RECOVERY_DEBOUNCE_SEC
                        ):
                            _sensor_error_latched = False

                    if _sensor_error_latched:
                        expected_light = "error"
                    else:
                        expected_light = "recording" if is_recording else "paused"

                    # 仅当状态发生变化时才执行更新 
                    light_apply_time = time.monotonic()
                    if (
                        expected_light != _current_light_state
                        or light_apply_time - _last_light_apply_time >= RGB_LIGHT_REFRESH_SEC
                    ):
                        if expected_light == "error":
                            rgb_indicator.set_error()
                        elif expected_light == "recording":
                            rgb_indicator.set_recording()
                        elif expected_light == "paused":
                            rgb_indicator.set_paused()
                        elif expected_light == "off":
                            rgb_indicator.turn_off_all()
                        _current_light_state = expected_light
                        _last_light_apply_time = light_apply_time

                    # ── STAGE 2: Get tactile camera data ────────────────────────────
                    left_color = tactile_frames_data.get("left", {}).get('color', np.zeros((224, 224, 3), dtype=np.uint8))
                    right_color = tactile_frames_data.get("right", {}).get('color', np.zeros((224, 224, 3), dtype=np.uint8))

                    # ── STAGE 3: Submit to UDP encoding queue ──────────────────────
                    # UDP preview uses a latest-frame policy: if network/encoding is slow,
                    # stale preview frames are dropped here without touching saved data.
                    current_frame_id = frame_id
                    frame_queued = False
                    try:
                        encode_item = (
                            wrist_frame.copy(),
                            left_color.copy(),
                            right_color.copy(),
                            metadata_data.copy() if isinstance(metadata_data, dict) else metadata_data,
                            current_frame_id,
                            time.time()
                        )
                        # 永远丢弃旧帧，强行塞入最新帧，保证图传极限低延迟
                        while True:
                            try:
                                encode_queue.put_nowait(encode_item)
                                break
                            except queue.Full:
                                try: 
                                    encode_queue.get_nowait()
                                    _dbg_encode_skip_count += 1
                                    _dbg_queue_full_count += 1
                                except queue.Empty: pass
                        
                        frame_queued = True
                        _dbg_encode_submit_count += 1
                        if _DEBUG_FRAME_LOG and (frame_id - _dbg_last_frame_print >= 5 or _dbg_last_frame_print == -1):
                            print(f"[MainLoop] ★ frame_id={frame_id} 入队成功(最新帧)! "
                                  f"sense={sense_time_ms:.1f}ms | encode_q={encode_queue.qsize()}/{ASYNC_QUEUE_SIZE}")
                            _dbg_last_frame_print = frame_id
                    except Exception as e:
                        print(f"[ERROR] encode_queue.put exception: {e}")

                    # Always increment frame_id to avoid duplicate IDs
                    frame_id += 1

                    # Adaptive sleep to prevent CPU overload
                    elapsed = time.time() - loop_start
                    sleep_time = max(0.001, _encode_interval - elapsed)

                    if _DEBUG_TIMING_LOG and _dbg_loop_count <= 100:
                        print(f"  └─ sense={sense_time_ms:.1f}ms | elapsed={elapsed*1000:.1f}ms | sleep={sleep_time*1000:.1f}ms")

                    time.sleep(sleep_time)

                    if check_esc_pressed():
                        print("\n[Main] ESC pressed. Exiting...")
                        break

            except Exception as e:
                print(f"\n[Main] Error captured: {e}")
                rgb_indicator.set_error() # 发生异常（红灯）
                import traceback
                traceback.print_exc()
            except KeyboardInterrupt:
                print("\n[Main] Ctrl-C caught. Exiting gracefully...")
                with recording_lock:
                    if is_recording:
                        print('[DEBUG] Final flush on manual exit! Saving remaining frames...')
                        env.get_and_save_data()
            finally:
                # ── Cleanup ──────────────────────────────────────────────────────────
                if 'rgb_indicator' in locals():
                    rgb_indicator.cleanup()
                env.stop_event.set()
                send_stop.set()
                # Clear encoding queue
                while not encode_queue.empty():
                    try:
                        encode_queue.get_nowait()
                    except queue.Empty:
                        break
                # Clear send queue
                while not send_queue.empty():
                    try:
                        send_queue.get_nowait()
                    except queue.Empty:
                        break
                save_thread.join(timeout=5.0)
                encode_thread.join(timeout=2.0)
                send_thread.join(timeout=2.0)
                try:
                    sock.close()
                except Exception:
                    pass
                try:
                    gpio_monitor.stop()
                except Exception:
                    pass
                print("[Main] Cleanup done, exiting...")
                
                # --- Time tracking summary ---
                if 'is_recording' in locals() and is_recording and 'current_episode_start' in locals() and current_episode_start is not None:
                    total_effective_time += time.time() - current_episode_start
                
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
                            f.write(f"Effective Data Time (seconds): {total_effective_time:.2f}\n")
                            f.write(f"Effective Data Time (formatted): {format_time(total_effective_time)}\n")
                            f.write(f"Total Episodes Recorded: {episode_index}\n")
                        print(f"\n[Summary] Saved time tracking to {summary_file}")
                    except Exception as e:
                        print(f"\n[Summary] Failed to save time tracking: {e}")


if __name__ == "__main__":
    init_keyboard()
    parser = ArgumentParser()
    parser.add_argument("--cap_id", type=int, default=112501) # monthday_times
    parser.add_argument("--fps", type=int, default=20)  # Reduced to 10 for stability
    parser.add_argument("--tactile_width", type=int, default=224)
    parser.add_argument("--tactile_height", type=int, default=224)
    parser.add_argument("--tactile_capture_width", type=int, default=640)
    parser.add_argument("--tactile_capture_height", type=int, default=480)
    args = parser.parse_args()
    main(args)
