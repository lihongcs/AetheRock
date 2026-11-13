# HandCap

HandCap records synchronized wrist RGB, tactile camera, gripper angle, and force data. The HTC scripts add tracker poses from a separate Windows host.

## Capture on Linux

Use a Linux or Raspberry Pi collector with Python 3.10+, Video4Linux devices, a GPIO button, I²C access for the gripper and force sensors, and the Hikrobot MVS SDK for the wrist camera. Install the Python dependencies for the collector in an isolated environment:

```bash
python -m pip install numpy opencv-python smbus pybullet threadpoolctl atomics
```

Configure the GPIO button and I²C addresses in `config.py`, or set the relevant environment variables before launching:

| Variable | Use |
| --- | --- |
| `HANDCAP_TACTILE_LEFT` / `HANDCAP_TACTILE_RIGHT` | Linux camera paths. Defaults are `/dev/video0` and `/dev/video2`. |
| `HANDCAP_WRIST_CAMERA_SERIAL` | MVS camera serial. It can be omitted when exactly one MVS camera is connected. |
| `HANDCAP_MVS_SDK_PATH` | Directory containing the SDK's `MvCameraControl_class.py` if it is outside the default SDK location and Python import path. |
| `HANDCAP_GPIO_NUM` | GPIO number of the capture button. The default is 364. |
| `HANDCAP_ANGLE_I2C_BUS`, `HANDCAP_FORCE_LEFT_I2C_BUS`, `HANDCAP_FORCE_RIGHT_I2C_BUS` | I²C buses for the gripper angle and force sensors. Defaults are 0, 1, and 2, respectively. |
| `HANDCAP_GRIPPER_CALIBRATION` | Path to the gripper calibration JSON. The default points to the calibration shipped with the code. |
| `HANDCAP_NTP_HOST` | NTP server for the optional synchronization check in `handcap_rgb.py`. |

From this directory, start the collector with a unique capture ID:

```bash
python handcap.py --cap_id 100001 --fps 20
```

Episodes are stored under `data/handcap_100001/episode_*`. For synchronized HTC poses, install `numpy`, `opencv-python`, `scipy`, `pyopenxr`, and `colorama` and configure the OpenXR runtime on the Windows host. Set `HANDCAP_RASPI_HOST` to the collector's IP address or hostname and run `python htc.py --cap_id 100001 --fps 20` with the same capture ID.

## Convert recordings to LeRobot

Use a Python 3.12 environment for the vendored LeRobot postprocessing code:

```bash
python -m pip install numpy opencv-python scipy pillow torch torchvision datasets huggingface-hub pyarrow pandas av fsspec packaging
```

The conversion scripts under `Postprocess/scripts/` take the same arguments as their Python entry points. To convert synchronized HandCap and HTC recordings, place both capture directories under the input root and run:

```bash
bash Postprocess/scripts/lerobot_no_phone.sh \
    --data_root ./data --output_root ./dataset --task "Example task" \
    --angle_calib_map /path/to/calibration-map.json
```

The example maps in `Postprocess/` use repository-relative paths. Set your own capture IDs and calibration files for newly collected data. The selected video codec must be available in the PyAV/FFmpeg installation.

To check the clock offset against a local NTP server, run `python Preprocess/_3_check_sync.py NTP_HOST`. You can also configure `HANDCAP_NTP_HOST` and omit the argument.

### iPad remote start for HTC

This matches the AnySense/Record3D USB style: USB is used through `iproxy`, which forwards a local TCP port to the iPad app over the cable. No Windows LAN IP is needed.

Run these on the Windows machine before starting collection from `instaumi`:

```bash
iproxy 8766 8766
```

In another terminal:

```bash
python htc_usb_bridge.py --port 8766
```

Then enable `Windows HTC 控制` in the iPad app and leave it in `USB 直连模式`.
When the iPad sends `总启动`, the bridge starts:

```bash
python htc_only.py --cap_id 112501 --collect-on-start
```

The same USB socket also returns Windows timestamps so the iPad can log the estimated `Windows - iPad` clock offset.

### iPad remote start for two HTC trackers

For the two-tracker setup (`right_elbow` and `left_elbow`), use the two-tracker bridge with the same USB port forwarding:

```bash
iproxy 8766 8766
```

In another terminal:

```bash
python htc_two_trackers_bridge.py --port 8766
```

or, for Ethernet/LAN:

```bat
start_lan_two_trackers_bridge.bat
```

This launches `htc_two_trackers.py` from the iPad control commands. Data is saved under `data/htc_only_CAPID/` as:

- `right_elbow.csv`
- `left_elbow.csv`

### Ethernet / LAN mode

If the iPad USB-C port is needed for a foot pedal or hub, connect iPad and Windows to the same Ethernet/LAN instead of using `iproxy`.

On the iPad:

1. Connect Ethernet or Wi-Fi.
2. Open `instaumi` -> `数据采集`.
3. Enable `Windows HTC 控制`.
4. Keep `USB/LAN 端口` as `8766`.
5. Find the iPad IP address in iPad Settings -> Ethernet/Wi-Fi details.

On Windows:

```bat
start_lan_bridge.bat
```

The script first tries to discover the iPad app through Bonjour/mDNS. Manual IP is still available as a fallback:

```bat
start_lan_bridge.bat IPAD_IP 8766
```

`IPAD_IP` is the iPad's Ethernet/Wi-Fi IP address, not the Windows IP.
For a direct Ethernet cable without a router, assign Windows and the iPad distinct addresses on the same subnet, then pass the iPad address to the launcher.

This uses the same command protocol and returns Windows timestamps for clock-offset logging.
