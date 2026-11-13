import os as _os

AUDIO_DEVICE_ID = int(_os.environ.get("HANDCAP_AUDIO_DEVICE_ID", "2"))
GPIO_NUM = int(_os.environ.get("HANDCAP_GPIO_NUM", "364"))
ANGLE_SENSOR = {
    "smbus_id": int(_os.environ.get("HANDCAP_ANGLE_I2C_BUS", "0")),
    "i2c_address": 0x36
}

FORCE_SENSOR = {
    "left": {"smbus_id": int(_os.environ.get("HANDCAP_FORCE_LEFT_I2C_BUS", "1")),
             "i2c_address": 0x48,},
    "right": {"smbus_id": int(_os.environ.get("HANDCAP_FORCE_RIGHT_I2C_BUS", "2")),
              "i2c_address": 0x48,}
}

TACTILE_CAMERA = {
    "left": _os.environ.get("HANDCAP_TACTILE_LEFT", "/dev/video0"),
    "right": _os.environ.get("HANDCAP_TACTILE_RIGHT", "/dev/video2"),
}

WRIST_CAMERA = {
    "wrist": {
        "serial": _os.environ.get("HANDCAP_WRIST_CAMERA_SERIAL"),
        "exposure": 20000,
    },
}
