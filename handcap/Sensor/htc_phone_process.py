import cv2
import json
import os
import time
import threading
import queue
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

from colorama import Fore, Style, init
init(autoreset=True)

@dataclass
class ChunkMetadata:
    chunk_id: int
    start_step: int
    end_step: int
    start_time: float
    end_time: float
    num_frames: int


def _has_valid_data(data) -> bool:
    """判断是否有有效数据"""
    if data is None:
        return False
    if isinstance(data, (dict, list, np.ndarray)) and len(data) == 0:
        return False
    return True

def _ndarray_to_list(obj):
    """递归将 ndarray 转换为 list，确保 JSON 可序列化"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _ndarray_to_list(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_ndarray_to_list(item) for item in obj]
    return obj


class LowLatencyRecorder:
    """
    低延迟传感器数据录制器
    HTC_pose / phone_pose -> JSON
    phone_camera           -> MP4 (h264)
    phone_depths     -> NPZ (压缩)
    """

    def __init__(
        self,
        save_dir: str,
        fps: int = 24,
        chunksize: int = 300,
        video_codec_priority: List[str] = None,
    ):
        self.save_dir = save_dir
        self.fps = fps
        self.chunksize = chunksize
        self.global_step = 0

        # 尝试使用更高效的编解码器
        if video_codec_priority is None:
            video_codec_priority = ['mp4v', 'h264']
        self._codec = self._select_codec(video_codec_priority)

        # ===== Ring Buffers =====
        self._htc_buffer = deque(maxlen=chunksize)
        self._phone_pose_buffer = deque(maxlen=chunksize)
        self._phone_camera_buffer = deque(maxlen=chunksize * 2)
        self._depth_buffer = deque(maxlen=chunksize)

        # ===== 写入队列 =====
        self._save_queue = queue.Queue(maxsize=4)
        self._running = True
        self._writer_thread = threading.Thread(target=self._save_worker, daemon=True)
        self._writer_thread.start()

        # ===== 当前录制状态 =====
        self._current_save_dir: Optional[str] = None
        self._chunk_counters: Dict[str, int] = {}
        self._chunk_metadata: List[ChunkMetadata] = []
        self._metadata_lock = threading.Lock()
        self._total_chunk_count = 0

        os.makedirs(save_dir, exist_ok=True)

    def _select_codec(self, priority: List[str]) -> int:
        fourcc_map = {
            'h264': cv2.VideoWriter_fourcc(*'h264'),
            'mp4v': cv2.VideoWriter_fourcc(*'mp4v'),
            'XVID': cv2.VideoWriter_fourcc(*'XVID'),
        }
        for codec in priority:
            if codec in fourcc_map:
                fourcc = fourcc_map[codec]
                test_path = os.path.join(self.save_dir, f'.codec_test_{codec}.mp4')
                try:
                    writer = cv2.VideoWriter(test_path, fourcc, 30, (224, 224))
                    if writer.isOpened():
                        writer.release()
                        os.remove(test_path)
                        print(f"[Recorder] Using video codec: {codec}")
                        return fourcc
                    writer.release()
                except Exception:
                    continue
        print("[Recorder] Falling back to mp4v codec")
        return cv2.VideoWriter_fourcc(*'mp4v')

    # ========== 主写入API ==========

    def put(
        self,
        htc_pose: Dict[str, Any] = None,
        phone_pose: Dict[str, Any] = None,
        phone_camera: Dict[str, Any] = None,
        phone_depth: Dict[str, Any] = None,
        save_dir: str = None,
        step: int = None,
    ):
        if step is None:
            step = self.global_step

        timestamp = time.time()

        # 检查是否切换目录
        # if save_dir != self._current_save_dir:
        #     if self._current_save_dir is not None:
        #         self._flush_all()
        #     self._current_save_dir = save_dir
        #     self._init_new_session(save_dir)
        # 检查是否切换目录
        if save_dir != self._current_save_dir:
            if self._current_save_dir is not None:
                self._flush_all_async()  # 改为异步，不阻塞主循环
            self._current_save_dir = save_dir
            self._init_new_session(save_dir)

        if save_dir is None:
            return

        # HTC Pose -> JSON buffer
        if _has_valid_data(htc_pose):
            self._htc_buffer.append({
                'step': step,
                'timestamp': timestamp,
                'data': htc_pose
            })

        # Phone Pose -> JSON buffer
        if _has_valid_data(phone_pose):
            self._phone_pose_buffer.append({
                'step': step,
                'timestamp': timestamp,
                'data': phone_pose
            })

        # Phone Camera -> MP4 buffer
        if _has_valid_data(phone_camera):
            self._phone_camera_buffer.append({
                'step': step,
                'timestamp': timestamp,
                'color': phone_camera.get('color'),
                'timestamp_arr': phone_camera.get('timestamp'),
                'step_idx': phone_camera.get('step_idx')
            })

        if _has_valid_data(phone_depth):
            self._depth_buffer.append({
                'step': step,
                'timestamp': timestamp,
                'data': phone_depth
            })

        # 检查是否需要自动保存chunk
        self._check_auto_save()

    def _check_auto_save(self):
        should_save = (
            len(self._htc_buffer) >= self.chunksize or
            len(self._phone_camera_buffer) >= self.chunksize
        )

        if should_save and self._current_save_dir is not None:
            self._queue_save_chunk()

    def _queue_save_chunk(self):
        if self._save_queue.full():
            return
        chunk_id = self._total_chunk_count
        self._total_chunk_count += 1

        chunk_data = {
            'chunk_id': chunk_id,
            'start_step': self.global_step - max(
                len(self._htc_buffer),
                len(self._phone_pose_buffer),
                len(self._phone_camera_buffer),
                len(self._depth_buffer)
            ),
            'htc_buffer': list(self._htc_buffer),
            'phone_pose_buffer': list(self._phone_pose_buffer),
            'phone_camera_buffer': list(self._phone_camera_buffer),
            'depth_buffer': list(self._depth_buffer),
            'save_dir': self._current_save_dir,
            'fps': self.fps,
            'codec': self._codec,
        }

        try:
            self._save_queue.put_nowait(chunk_data)
            self._htc_buffer.clear()
            self._phone_pose_buffer.clear()
            self._phone_camera_buffer.clear()
            self._depth_buffer.clear()
        except queue.Full:
            pass

    # def _flush_all(self):
    #     if len(self._htc_buffer) > 0 or len(self._phone_camera_buffer) > 0:
    #         self._queue_save_chunk()
    #     elif len(self._depth_buffer) > 0:
    #         self._queue_save_chunk()

    #     self._save_queue.join()
    def _flush_all(self):
        # 将当前缓冲数据全部推入写入队列，立即返回，不阻塞主循环
        if len(self._htc_buffer) > 0 or len(self._phone_camera_buffer) > 0:
            self._queue_save_chunk()
        elif len(self._depth_buffer) > 0:
            self._queue_save_chunk()

    # def stop(self, save_remaining: bool = True):
    #     self._running = False

    #     if save_remaining:
    #         self._flush_all()

    #     self._save_metadata_index()
    #     self._writer_thread.join(timeout=5.0)
    def stop(self, save_remaining: bool = True):
        self._running = False
        if save_remaining:
            self._flush_all_async()  # 改为异步，退出时立即返回
        self._metadata_lock.acquire()
        try:
            self._metadata_index_dirty = True
        finally:
            self._metadata_lock.release()
        self._writer_thread.join(timeout=2.0)  # 最多等 2s，不再无限等待

    # ========== 后台写入线程 ==========

    def _save_worker(self):
        while self._running or not self._save_queue.empty():
            try:
                chunk = self._save_queue.get(timeout=0.1)
                try:
                    self._save_chunk(chunk)
                except Exception as e:
                    import traceback
                    print(f"[Recorder] _save_chunk error: {e}")
                    traceback.print_exc()
                self._save_queue.task_done()
            except queue.Empty:
                continue

    def _save_chunk(self, chunk: dict):
        chunk_id = chunk['chunk_id']
        save_dir = chunk['save_dir']
        fps = chunk['fps']
        codec = chunk['codec']

        start_time = time.time()

        # 1. HTC Pose -> JSON
        if chunk['htc_buffer']:
            self._save_json_chunk(
                save_dir=os.path.join(save_dir, 'htc_pose'),
                buffer=chunk['htc_buffer'],
                prefix='htc_pose',
                chunk_id=chunk_id
            )

        # 2. Phone Pose -> JSON
        if chunk['phone_pose_buffer']:
            self._save_json_chunk(
                save_dir=os.path.join(save_dir, 'phone_pose'),
                buffer=chunk['phone_pose_buffer'],
                prefix='phone_pose',
                chunk_id=chunk_id
            )

        # 3. Phone Camera -> MP4
        if chunk['phone_camera_buffer']:
            self._save_video_chunk(
                save_dir=os.path.join(save_dir, 'phone_camera'),
                buffer=chunk['phone_camera_buffer'],
                prefix='phone_camera',
                chunk_id=chunk_id,
                fps=fps,
                codec=codec
            )

        # 4. Depth -> NPZ（仅当有有效数据）
        if chunk['depth_buffer']:
            first_data = chunk['depth_buffer'][0]['data']
            if _has_valid_data(first_data) and first_data.get('depth') is not None:
                self._save_depth_chunk(
                    save_dir=os.path.join(save_dir, 'phone_depth'),
                    buffer=chunk['depth_buffer'],
                    prefix='depth',
                    chunk_id=chunk_id
                )

        # 记录元数据
        with self._metadata_lock:
            valid_buffer = chunk['htc_buffer'] or chunk['phone_pose_buffer'] or \
                          chunk['phone_camera_buffer'] or chunk['depth_buffer']
            if valid_buffer:
                timestamps = [item['timestamp'] for item in valid_buffer]
                self._chunk_metadata.append(ChunkMetadata(
                    chunk_id=chunk_id,
                    start_step=chunk['start_step'],
                    end_step=chunk['start_step'] + len(valid_buffer),
                    start_time=timestamps[0] if timestamps else 0,
                    end_time=timestamps[-1] if timestamps else 0,
                    num_frames=len(valid_buffer)
                ))

        duration = time.time() - start_time
        print(f"[Recorder] Chunk {chunk_id} saved in {duration:.3f}s")

    # ========== 各类数据保存方法 ==========

    def _save_json_chunk(self, save_dir: str, buffer: list, prefix: str, chunk_id: int):
        os.makedirs(save_dir, exist_ok=True)

        if not buffer:
            return

        save_data = {
            'chunk_id': chunk_id,
            'num_samples': len(buffer),
            'timestamps': [item['timestamp'] for item in buffer],
            'steps': [item['step'] for item in buffer],
            'data': [_ndarray_to_list(item['data']) for item in buffer],
        }

        path = os.path.join(save_dir, f'{prefix}_chunk_{chunk_id:06d}.json')
        with open(path, 'w') as f:
            json.dump(save_data, f)

    def _save_video_chunk(
        self,
        save_dir: str,
        buffer: list,
        prefix: str,
        chunk_id: int,
        fps: int,
        codec
    ):
        os.makedirs(save_dir, exist_ok=True)

        if not buffer:
            return

        first_frame = buffer[0]['color']
        if len(first_frame.shape) == 4:
            h, w = first_frame.shape[1:3]
        elif len(first_frame.shape) == 3:
            h, w = first_frame.shape[:2]
        else:
            h, w = 480, 640

        video_path = os.path.join(save_dir, f'{prefix}_chunk_{chunk_id:06d}.mp4')
        writer = cv2.VideoWriter(video_path, codec, fps, (w, h))

        if not writer.isOpened():
            print(f"[WARN] VideoWriter failed to open: {video_path}")
            return

        all_timestamps = []
        all_step_idxs = []

        for item in buffer:
            frames = item['color']

            if len(frames.shape) == 4:
                for frame in frames:
                    writer.write(frame)
            elif len(frames.shape) == 3:
                writer.write(frames)

            if item.get('timestamp_arr') is not None:
                ts = item['timestamp_arr']
                if hasattr(ts, 'tolist'):
                    all_timestamps.extend(ts.tolist())
                elif isinstance(ts, (list, tuple)):
                    all_timestamps.extend(ts)
                else:
                    all_timestamps.append(ts)
            if item.get('step_idx') is not None:
                si = item['step_idx']
                if hasattr(si, 'tolist'):
                    all_step_idxs.extend(si.tolist())
                elif isinstance(si, (list, tuple)):
                    all_step_idxs.extend(si)
                else:
                    all_step_idxs.append(si)

        writer.release()

        index_path = os.path.join(save_dir, f'{prefix}_chunk_{chunk_id:06d}.json')
        with open(index_path, 'w') as f:
            json.dump({
                'chunk_id': chunk_id,
                'video_path': video_path,
                'fps': fps,
                'timestamps': all_timestamps,
                'step_idxs': all_step_idxs
            }, f)

    def _save_depth_chunk(self, save_dir, buffer, prefix, chunk_id):
        os.makedirs(save_dir, exist_ok=True)

        if not buffer:
            return

        steps = []
        timestamps = []
        depth_arrays = []

        for item in buffer:
            data = item['data']
            steps.append(data['step_idx'])
            timestamps.append(data['timestamp'])
            depth_arrays.append(data['depth'])

        # 1. 保存整个 chunk 为一个 .npz 文件
        npz_path = os.path.join(save_dir, f'{prefix}_chunk_{chunk_id:06d}.npz')
        np.savez_compressed(npz_path, depths=np.stack(depth_arrays, axis=0))

        # 2. 保存 metadata.json（timestamps + steps）
        meta_path = os.path.join(save_dir, f'{prefix}_chunk_{chunk_id:06d}.json')
        with open(meta_path, 'w') as f:
            json.dump({
                'chunk_id': chunk_id,
                'num_frames': len(buffer),
                'steps': steps,
                'shapes': [arr.shape for arr in depth_arrays],
                'timestamps': timestamps,
            }, f)

    def _save_metadata_index(self):
        save_dir = self._current_save_dir or self.save_dir
        index_path = os.path.join(save_dir, 'recording_metadata.json')

        with self._metadata_lock:
            metadata = {
                'total_chunks': len(self._chunk_metadata),
                'fps': self.fps,
                'chunks': [
                    {
                        'chunk_id': m.chunk_id,
                        'start_step': m.start_step,
                        'end_step': m.end_step,
                        'num_frames': m.num_frames,
                        'start_time': m.start_time,
                        'end_time': m.end_time,
                    }
                    for m in self._chunk_metadata
                ]
            }

        with open(index_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def _init_new_session(self, save_dir: str):

        self._htc_buffer.clear()
        self._phone_pose_buffer.clear()
        self._phone_camera_buffer.clear()
        self._depth_buffer.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
    
    def _flush_all_async(self):
        """非阻塞版本：将缓冲数据推入队列，不等待写入完成"""
        if len(self._htc_buffer) > 0 or len(self._phone_camera_buffer) > 0:
            self._queue_save_chunk()
        elif len(self._depth_buffer) > 0:
            self._queue_save_chunk()
        # 主循环立即继续，worker 线程在后台写入


# ========== 集成到你的环境 ==========

class HTCPhoneDataRecorder:
    def __init__(
        self,
        save_dir: str,
        fps: int = 24,
        chunksize: int = 300,
        camera_resolution: tuple = (224, 224)
    ):
        self.save_dir = save_dir
        self.fps = fps
        self.camera_resolution = camera_resolution
        self._recorder: Optional[LowLatencyRecorder] = None
        self._chunksize = chunksize

    def start(self):
        self._recorder = LowLatencyRecorder(
            save_dir=self.save_dir,
            fps=self.fps,
            chunksize=self._chunksize
        )

    def put(
        self,
        htc_data: Dict = None,
        phone_pose: Dict = None,
        phone_camera: Dict = None,
        phone_depth: Dict = None,
        save_dir: str = None,
        step: int = None,
    ):
        if self._recorder is None:
            return

        self._recorder.put(
            htc_pose=htc_data,
            phone_pose=phone_pose,
            phone_camera=phone_camera,
            phone_depth=phone_depth,
            save_dir=save_dir,
            step=step
        )

    def stop(self):
        # [DEBUG ac947b] log stop called
       if self._recorder:
            self._recorder.stop()
            self._recorder = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()