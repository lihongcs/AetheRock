from __future__ import annotations

from abc import ABC, abstractmethod
import json
from pathlib import Path
from typing import Any


class BaseCamera(ABC):
    camera_registry: dict[str, type] = {}

    @abstractmethod
    def __init__(self, camera_id: str):
        pass

    @abstractmethod
    def get_data(self):
        pass

    @classmethod
    def register_camera(cls, camera_name: str):
        def wrapper(subclass: type) -> type:
            cls.camera_registry[camera_name] = subclass
            return subclass

        return wrapper

    @classmethod
    def create_camera(
        cls,
        camera_name: str,
        camera_sn: str | None = None,
        camera_v4l_path: str | None = None,
        fps: int | None = None,
        resolution: Any | None = None,
        capture_resolution: Any | None = None,
    ):
        if camera_name == "realsense":
            from perception.cameras import realsense  # noqa: F401
        if camera_name in {"opencv", "tactile_opencv"}:
            from perception.cameras import tactile_opencv  # noqa: F401
        if camera_name in {"insta_front_fisheye", "insta360_front_fisheye"}:
            from perception.cameras import insta_front_fisheye  # noqa: F401

        if camera_name not in cls.camera_registry:
            raise ValueError(f"Camera with key '{camera_name}' is not registered.")

        camera_class = cls.camera_registry[camera_name]
        if camera_name == "realsense":
            if not camera_sn:
                raise ValueError("RealSense camera config requires a non-empty 'sn'.")
            return camera_class(camera_sn)

        if camera_name in {"opencv", "tactile_opencv"}:
            return camera_class(camera_sn, camera_v4l_path, fps, resolution, capture_resolution)

        if camera_name in {"insta_front_fisheye", "insta360_front_fisheye"}:
            return camera_class(camera_sn, fps=fps, resolution=resolution, capture_resolution=capture_resolution)

        return camera_class(camera_sn, camera_v4l_path, fps, resolution)

    @classmethod
    def create_cameras_from_config(cls, config_path: str):
        with open(Path(config_path).expanduser(), "r", encoding="utf-8") as f:
            config = json.load(f)

        camera_dict = {}
        for view, camera_info in config.items():
            camera_type = camera_info["type"]
            if camera_type == "realsense":
                camera = cls.create_camera(camera_type, camera_info.get("sn"))
            elif camera_type == "mvs_cam":
                from perception.cameras.mvs_cam import MVSCamera

                camera = MVSCamera(
                    {
                        "serial": camera_info.get("serial"),
                        "exposure": camera_info.get("exposure"),
                    }
                )
            else:
                camera = cls.create_camera(
                    camera_type,
                    camera_info.get("camera_index"),
                    camera_info.get("camera_v4l_path"),
                    camera_info.get("fps"),
                    camera_info.get("resolution"),
                    camera_info.get("capture_resolution"),
                )
            camera_dict[view] = camera

        return camera_dict
