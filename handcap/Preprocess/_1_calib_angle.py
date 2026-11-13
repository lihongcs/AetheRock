"""Calibrate the angle sensor (AS5600) with gripper width.
"""
import os
import time
import json
import smbus

from argparse import ArgumentParser


DEVICE_AS5600 = 0x36
bus = smbus.SMBus(0)

def ReadRawAngle():
    try:
        read_bytes = bus.read_i2c_block_data(DEVICE_AS5600, 0x0C, 2)
        return (read_bytes[0] << 8) | read_bytes[1]
    except Exception as e:
        print(f"Failed to read sensor: {e}")
        return None

def calibrate_gripper(args):
    
    if os.path.exists(f"angle_calib_params/gripper_calibration_interpolation{args.cap_id}.json"):
        print(f"警告：校准文件 angle_calib_params/gripper_calibration_interpolation{args.cap_id}.json 已存在，可能会被覆盖。")
        raise FileExistsError(f"校准文件 angle_calib_params/gripper_calibration_interpolation{args.cap_id}.json 已存在，可能会被覆盖。")
    
    calibration_data = {
        "metadata": {
            "unit": "cm",
            "sensor_range": [0, 4095],
        },
        "calibration": []
    }
    
    print("=== Calibration of gripper width and as5600 readings ===")

    
    for width in range(8, -1, -1): # 8cm -> 1cm
        while True:
            user_input = input(f"Adjust your gripper width to {width}cm, and press ENTER to record: ")
            readings = []
            for _ in range(5):
                if angle := ReadRawAngle():
                    readings.append(angle)
                    time.sleep(0.05)
            
            avg_angle = sum(readings) // len(readings)
            calibration_data["calibration"].append( (width / 100.0, avg_angle) )  # meter

            print(f"Recorded: width {width}cm -> angle value {avg_angle}")
            break

    os.makedirs("angle_calib_params", exist_ok=True)
    
    with open(f"angle_calib_params/gripper_calibration_interpolation{args.cap_id}.json", "w") as f:
        json.dump(calibration_data, f, indent=2)
    
    print(f"\nSaved to angle_calib_params/gripper_calibration_interpolation{args.cap_id}.json")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--cap_id", type=str, default="110303") # monthday
    args = parser.parse_args()
    calibrate_gripper(args)