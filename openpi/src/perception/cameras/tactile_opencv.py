from __future__ import annotations

import platform
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .base_camera import BaseCamera


@BaseCamera.register_camera("tactile_opencv")
@BaseCamera.register_camera("opencv")
class TactileOpenCVCamera(BaseCamera):
    def __init__(
        self,
        camera_index: str | int | None = None,
        camera_v4l_path: str | None = None,
        fps: int | None = None,
        resolution: Any | None = None,
        capture_resolution: Any | None = None,
    ) -> None:
        self.camera_index = camera_index
        self.camera_v4l_path = camera_v4l_path
        self.fps = int(fps) if fps is not None else 20
        self.resolution = self._parse_resolution(resolution or (224, 224))
        self.capture_resolution = self._parse_resolution(capture_resolution or (640, 480))
        self.device = self._resolve_device()
        if platform.system() == "Linux":
            self.capture = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        else:
            self.capture = cv2.VideoCapture(self.device)
        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open tactile OpenCV camera: {self.device}")

        self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
        capture_w, capture_h = self.capture_resolution
        self.capture.set(cv2.CAP_PROP_FPS, self.fps)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, capture_w)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, capture_h)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def _resolve_device(self) -> str | int:
        if self.camera_v4l_path:
            return self.camera_v4l_path
        if self.camera_index is None:
            raise ValueError("tactile_opencv camera requires camera_index or camera_v4l_path")
        if isinstance(self.camera_index, str) and self.camera_index.startswith("/"):
            return self.camera_index
        if isinstance(self.camera_index, str) and self.camera_index.isdigit():
            self.camera_index = int(self.camera_index)
        if platform.system() == "Linux":
            return str(Path("/dev") / f"video{self.camera_index}")
        return int(self.camera_index)

    @staticmethod
    def _parse_resolution(resolution: Any) -> tuple[int, int]:
        if isinstance(resolution, str):
            parts = resolution.lower().replace("x", ",").split(",")
            if len(parts) != 2:
                raise ValueError(f"Invalid resolution string: {resolution!r}")
            return int(parts[0]), int(parts[1])
        if isinstance(resolution, (list, tuple)) and len(resolution) == 2:
            return int(resolution[0]), int(resolution[1])
        raise ValueError(f"Invalid resolution: {resolution!r}")

    def read(self):
        timestamp = time.time()
        ok, frame_bgr = self.capture.read()
        if not ok or frame_bgr is None:
            return False, None, timestamp
        if (frame_bgr.shape[1], frame_bgr.shape[0]) != self.resolution:
            frame_bgr = cv2.resize(frame_bgr, self.resolution, interpolation=cv2.INTER_AREA)
        return True, frame_bgr, timestamp

    def get_data(self, return_type: str = "bgr"):
        ok, frame_bgr, timestamp = self.read()
        if not ok or frame_bgr is None:
            return None, timestamp
        if return_type == "bgr":
            return frame_bgr, timestamp
        if return_type == "rgb":
            return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), timestamp
        raise ValueError(f"Unsupported return_type: {return_type}")

    def release(self) -> None:
        if getattr(self, "capture", None) is not None:
            self.capture.release()
            self.capture = None

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass
