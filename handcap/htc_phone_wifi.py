"""
AnySense Wi-Fi / USB Receiver
Connects directly to the AnySense app over Wi-Fi or via USB (iproxy)
and displays the RGB/Depth stream.

Usage (Wi-Fi):
  1. Connect both devices to the same network.
  2. On iPhone: AnySense app -> Settings -> Wi-Fi mode -> press Record button.
  3. Note the IP displayed on the screen.
  4. Run: python htc_phone_wifi.py --host <IP_ADDRESS>

Usage (USB):
  1. Run: iproxy 1337 1337  (keep running in another terminal)
  2. On iPhone: AnySense app -> Settings -> USB mode -> press Record button
  3. Run: python htc_phone_wifi.py --host 127.0.0.1
"""

import socket
import struct
import numpy as np
import cv2
import zlib
import argparse

import lzfse

PORT = 1337

# PeerTalkHeader: 4 x UInt32. Swift stores a,b,c as little-endian, body_size as big-endian.
# Total = 16 bytes
PEERTALK_HEADER_SIZE = 16

# Record3DHeader: 11 x UInt32 (little-endian, native ARM64)
# rgbWidth, rgbHeight, depthWidth, depthHeight, confWidth, confHeight,
# rgbSize, depthSize, confMapSize, miscSize, deviceType
RECORD3D_HEADER_SIZE = 44  # 11 * 4

# IntrinsicMatrixCoeffs: 4 x Float32 = 16 bytes
INTRINSIC_SIZE = 16

# CameraPose: 7 x Float32 (qx,qy,qz,qw, tx,ty,tz) = 28 bytes
POSE_SIZE = 28


def recv_exact(sock, n):
    data = b''
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise ConnectionResetError("Connection closed by server")
        data += chunk
    return data


def main(args):
    HOST = args.host
    if HOST == '0.0.0.0':
        HOST = input("Please enter the Wi-Fi IP address shown on your iPhone: ")
        
    print(f"Connecting to AnySense at {HOST}:{PORT}...")

    while True:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(3.0) # connection timeout
                s.connect((HOST, PORT))
                s.settimeout(None) # disable timeout for blocking recv
                print(f"✅ Connected to {HOST}! Waiting for stream data (Make sure to press Record)...")

                frame_count = 0
                while True:
                    try:
                        # --- 1. PeerTalk header (16 bytes) ---
                        pt_raw = recv_exact(s, PEERTALK_HEADER_SIZE)
                        # a, b, c are little-endian UInt32; body_size is big-endian UInt32
                        pt_a, pt_b, pt_c = struct.unpack_from('<III', pt_raw, 0)
                        body_size = struct.unpack_from('>I', pt_raw, 12)[0]

                        if body_size == 0:
                            continue

                        # --- 2. Full body ---
                        body = recv_exact(s, body_size)
                        offset = 0

                        # --- 3. Record3D header (44 bytes, little-endian) ---
                        if len(body) < offset + RECORD3D_HEADER_SIZE:
                            print(f"Body too short for Record3D header: {len(body)}")
                            continue
                        (rgb_w, rgb_h, depth_w, depth_h, conf_w, conf_h,
                         rgb_size, depth_size, conf_size, misc_size, device_type
                         ) = struct.unpack_from('<11I', body, offset)
                        offset += RECORD3D_HEADER_SIZE

                        # --- 4. Intrinsic matrix (16 bytes) ---
                        if len(body) < offset + INTRINSIC_SIZE:
                            continue
                        fx, fy, tx, ty = struct.unpack_from('<4f', body, offset)
                        offset += INTRINSIC_SIZE

                        # --- 5. Camera pose (28 bytes) ---
                        if len(body) < offset + POSE_SIZE:
                            continue
                        qx, qy, qz, qw, tx_p, ty_p, tz_p = struct.unpack_from('<7f', body, offset)
                        offset += POSE_SIZE

                        # --- 6. RGB JPEG data ---
                        if rgb_size == 0:
                            print(f"Warning: rgb_size is 0! body={len(body)}, offset={offset}")
                            continue
                        if len(body) < offset + rgb_size:
                            print(f"Body too short for RGB: need {offset + rgb_size}, have {len(body)}")
                            continue

                        rgb_jpeg = body[offset:offset + rgb_size]
                        offset += rgb_size

                        # Decode JPEG
                        nparr = np.frombuffer(rgb_jpeg, np.uint8)
                        rgb_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                        # --- 7. Depth data ---
                        depth_color = None

                        if depth_size > 0 and len(body) >= offset + depth_size:
                            depth_compressed = body[offset:offset + depth_size]
                            offset += depth_size

                            try:
                                depth_raw = lzfse.decompress(depth_compressed)
                            except Exception:
                                depth_raw = depth_compressed  # fallback

                            raw_len = len(depth_raw)
                            expected = depth_w * depth_h * 4

                            if raw_len >= expected:
                                depth_arr = np.frombuffer(depth_raw, np.float32).copy().reshape((depth_h, depth_w))
                            else:
                                print(f"Warning: depth raw_len={raw_len}, expected>={expected}")
                                depth_arr = None

                            if depth_arr is not None:
                                v = depth_arr.ravel()
                                v = v[v > 0.001]
                                v_min = np.percentile(v, 2) if v.size > 10 else depth_arr.min()
                                v_max = np.percentile(v, 98) if v.size > 10 else depth_arr.max()
                                if v_max > v_min:
                                    depth_norm = ((depth_arr - v_min) / (v_max - v_min) * 255).clip(0, 255).astype(np.uint8)
                                else:
                                    depth_norm = np.zeros(depth_arr.shape, dtype=np.uint8)
                                depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_TURBO)

                        if rgb_img is not None:
                            frame_count += 1
                            depth_info = f"Depth: {depth_w}x{depth_h} " if depth_color is not None else "No depth "
                            print(f"Frame {frame_count:4d} | {rgb_w}x{rgb_h} | "
                                  f"Pose: ({tx_p:.2f}, {ty_p:.2f}, {tz_p:.2f}) | "
                                  f"{depth_info}| RGB: {rgb_size} bytes", end='\r')

                            # 组合 RGB 和 Depth 显示
                            if depth_color is not None:
                                rgb_display = rgb_img
                                depth_resized = cv2.resize(depth_color, (rgb_w, rgb_h))
                                combined = np.hstack((rgb_display, depth_resized))

                                rgb_label = np.zeros((40, rgb_w, 3), dtype=np.uint8)
                                depth_label = np.zeros((40, rgb_w, 3), dtype=np.uint8)
                                cv2.putText(rgb_label, "RGB Stream", (10, 28),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                                cv2.putText(depth_label, "Depth Map", (10, 28),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                                labels = np.hstack((rgb_label, depth_label))
                                combined = np.vstack((labels, combined))

                                cv2.imshow('AnySense - RGB & Depth', combined)
                            else:
                                cv2.imshow('AnySense RGB', rgb_img)

                            key = cv2.waitKey(1) & 0xFF
                            if key == ord('q'):
                                print("\nQuit requested.")
                                cv2.destroyAllWindows()
                                return
                        else:
                            print(f"Failed to decode JPEG (size={rgb_size}, offset at start={offset - rgb_size})")

                    except ConnectionResetError:
                        print("\n⚠️ Connection closed by phone. Will retry in 2 seconds...")
                        break
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        break
        except (ConnectionRefusedError, socket.timeout, OSError) as e:
            print(f"Waiting for connection: {e} (Is the app open and in Wi-Fi mode?)")
            
        import time
        time.sleep(2)

    cv2.destroyAllWindows()
    print("Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="AnySense RGB-D Receiver")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="iPhone IP address (or 127.0.0.1 for USB iproxy)")
    args = parser.parse_args()
    main(args)
