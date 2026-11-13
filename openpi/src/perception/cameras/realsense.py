from __future__ import annotations

import logging
import time

import cv2
import numpy as np
from PIL import Image

from .base_camera import BaseCamera


logger = logging.getLogger(__name__)


@BaseCamera.register_camera("realsense")
class RealSense(BaseCamera):
    def __init__(self, camera_sn: str):
        import pyrealsense2 as rs

        self.rs = rs
        self.camera_sn = str(camera_sn)
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.pipeline_started = False

        connected = [device.get_info(rs.camera_info.serial_number) for device in rs.context().query_devices()]
        if self.camera_sn not in connected:
            raise RuntimeError(
                f"RealSense serial {self.camera_sn} was not found. Connected RealSense devices: {connected}"
            )

        self.config.enable_device(self.camera_sn)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.align = rs.align(rs.stream.color)
        self.pipeline_profile = self.pipeline.start(self.config)
        self.pipeline_started = True
        self.hole_filling = rs.hole_filling_filter()
        self.intrinsics = self.getIntrinsics()

        logger.info("Warming up RealSense camera %s ...", self.camera_sn)
        warmup = 60
        while warmup > 0:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            if aligned_frames.get_depth_frame() and aligned_frames.get_color_frame():
                warmup -= 1
        logger.info("RealSense camera %s initialized.", self.camera_sn)

    def _read_color_bgr(self) -> np.ndarray:
        while True:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            if not color_frame:
                continue
            return np.asanyarray(color_frame.get_data())

    def read(self):
        color_bgr = self._read_color_bgr()
        return True, color_bgr, time.time()

    def get_data(self, hole_filling: bool = False, return_type: str = "PIL"):
        color_bgr = self._read_color_bgr()
        if return_type == "bgr":
            return color_bgr, None

        color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        if return_type == "PIL":
            return Image.fromarray(color_rgb), None
        if return_type == "numpy":
            return color_rgb, None
        raise ValueError(f"Unsupported return_type: {return_type}")

    def getIntrinsics(self):
        color_frame = None
        for _ in range(60):
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            if color_frame:
                break
        if not color_frame:
            raise RuntimeError(f"Could not read color intrinsics from RealSense camera {self.camera_sn}.")

        intrinsics = color_frame.get_profile().as_video_stream_profile().get_intrinsics()
        return np.array(
            [
                [intrinsics.fx, 0.0, intrinsics.ppx],
                [0.0, intrinsics.fy, intrinsics.ppy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def release(self) -> None:
        if self.pipeline_started:
            self.pipeline.stop()
            self.pipeline_started = False

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass
