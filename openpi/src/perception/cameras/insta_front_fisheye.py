from __future__ import annotations

import logging
import os
from pathlib import Path
import shlex
import signal
import subprocess
import threading
import time
from typing import Any

import cv2
import numpy as np

from .base_camera import BaseCamera


logger = logging.getLogger(__name__)


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@BaseCamera.register_camera("insta360_front_fisheye")
@BaseCamera.register_camera("insta_front_fisheye")
class InstaFrontFisheyeCamera(BaseCamera):
    def __init__(
        self,
        serial: str | None = None,
        *,
        fps: int | None = None,
        resolution: Any | None = None,
        capture_resolution: Any | None = None,
    ) -> None:
        self.serial = os.environ.get("INSTA_SERIAL") or (str(serial) if serial else "")
        if not self.serial:
            raise ValueError("insta_front_fisheye requires serial in camera config or INSTA_SERIAL")

        env_fps = os.environ.get("INSTA_FRONT_FPS")
        self.fps = int(env_fps) if env_fps else int(fps or 20)
        self.resolution = self._parse_resolution(os.environ.get("INSTA_FRONT_RESOLUTION") or resolution or (224, 224))
        self.capture_resolution = self._parse_resolution(
            os.environ.get("INSTA_FRONT_CAPTURE_RESOLUTION") or capture_resolution or self.resolution
        )
        self.frame_bytes = self.capture_resolution[0] * self.capture_resolution[1] * 3

        binary = os.environ.get("INSTA_CAMERA_BINARY")
        if not binary:
            raise ValueError("Set INSTA_CAMERA_BINARY to the path of the Insta camera capture binary.")
        self.binary_path = Path(binary).expanduser()
        if not self.binary_path.exists():
            raise FileNotFoundError(
                f"Insta camera binary not found: {self.binary_path}."
            )

        self.process: subprocess.Popen[bytes] | None = None
        self._stderr_tail: list[str] = []
        self._stderr_thread: threading.Thread | None = None
        self._primed_frame: tuple[np.ndarray, float] | None = None

        self._start_process()
        warmup_frames = int(os.environ.get("INSTA_WARMUP_FRAMES", "1"))
        for _ in range(max(0, warmup_frames)):
            self._primed_frame = self._read_frame()

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

    def _build_command(self) -> list[str]:
        capture_w, capture_h = self.capture_resolution
        cmd = [
            str(self.binary_path),
            "--serial",
            self.serial,
            "--front-raw-stdout",
            "--raw-width",
            str(capture_w),
            "--raw-height",
            str(capture_h),
            "--raw-fps",
            str(self.fps),
        ]
        if os.environ.get("INSTA_RES"):
            cmd.extend(["--res", os.environ["INSTA_RES"]])
        if _env_flag("INSTA_NO_STITCH", False):
            cmd.append("--no-stitch")
        if os.environ.get("INSTA_SERVICE_PORT"):
            cmd.extend(["--service-port", os.environ["INSTA_SERVICE_PORT"]])
        if os.environ.get("INSTA_CAMERA_EXTRA_ARGS"):
            cmd.extend(shlex.split(os.environ["INSTA_CAMERA_EXTRA_ARGS"]))
        return cmd

    def _start_process(self) -> None:
        cmd = self._build_command()
        logger.info(
            "Starting Insta front fisheye camera: serial=%s raw=%dx%d fps=%d cmd=%s",
            self.serial,
            self.capture_resolution[0],
            self.capture_resolution[1],
            self.fps,
            " ".join(shlex.quote(part) for part in cmd),
        )
        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            start_new_session=True,
        )
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True, name="insta-front-fisheye-stderr")
        self._stderr_thread.start()

    def _drain_stderr(self) -> None:
        process = self.process
        if process is None or process.stderr is None:
            return
        for raw_line in iter(process.stderr.readline, b""):
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            self._stderr_tail = (self._stderr_tail + [line])[-80:]
            logger.info("Insta camera: %s", line)

    def _process_error(self, detail: str) -> RuntimeError:
        process = self.process
        returncode = process.poll() if process is not None else None
        tail = "\n".join(self._stderr_tail[-20:])
        suffix = f" stderr tail:\n{tail}" if tail else ""
        return RuntimeError(f"Insta front fisheye stream failed: {detail}; returncode={returncode}.{suffix}")

    def _read_exact(self, size: int) -> bytes:
        process = self.process
        if process is None or process.stdout is None:
            raise self._process_error("process is not running")

        chunks: list[bytes] = []
        remaining = size
        while remaining > 0:
            chunk = process.stdout.read(remaining)
            if not chunk:
                raise self._process_error(f"EOF while reading raw frame, missing {remaining} bytes")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _read_frame(self) -> tuple[np.ndarray, float]:
        data = self._read_exact(self.frame_bytes)
        timestamp = time.time()
        capture_w, capture_h = self.capture_resolution
        frame = np.frombuffer(data, dtype=np.uint8).reshape((capture_h, capture_w, 3)).copy()
        if self.resolution != self.capture_resolution:
            frame = cv2.resize(frame, self.resolution, interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(frame), timestamp

    def read(self):
        if self._primed_frame is not None:
            frame, timestamp = self._primed_frame
            self._primed_frame = None
            return True, frame.copy(), timestamp
        frame, timestamp = self._read_frame()
        return True, frame, timestamp

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
        process = self.process
        self.process = None
        if process is None:
            return

        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=2.0)

        for stream in (process.stdout, process.stderr):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass
