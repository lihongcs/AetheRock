import os
import cv2
import time
import json
import numpy as np
from pyzbar import pyzbar
from argparse import ArgumentParser


def run_qr_scanner(camera_index=0, time_difference_name="1127", output_filename="1127.mp4"):
    """
    启动二维码扫描器，支持读取完整时间戳或"极简模式"（后6位）时间戳，
    并计算端到端延迟。
    """
    
    if os.path.exists(time_difference_name):
        print(f"警告：时间差文件 {time_difference_name} 已存在，可能会被覆盖。")
        raise FileExistsError(f"时间差文件 {time_difference_name} 已存在，可能会被覆盖。")
    
    os.makedirs("calibrated_time_difference", exist_ok=True)
    camera = cv2.VideoCapture(camera_index)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1) 
    
    if not camera.isOpened():
        print(f"错误：无法打开摄像头索引 {camera_index}。")
        return

    frame_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 30.0 

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))
    
    print(f"摄像头已启动 (分辨率: {frame_width}x{frame_height})")
    print("支持读取：完整时间戳 或 极简时间戳(后6位)")
    print("按 'q' 键退出...")

    os.makedirs("checked_images", exist_ok=True)
    
    last_decoded_data = None
    total_time_difference = []
    
    # 用于防止保存图片过于频繁
    frame_count = 0 
    
    while True:
        # 1. 获取当前扫描器的基准时间
        scanner_now = time.time()
        scanner_time_ms = int(scanner_now * 1000)
        
        ret, frame = camera.read()
        if not ret:
            print("错误：无法读取视频帧。")
            break
            
        frame_count += 1
        
        if frame_count % 20 == 0:
            output_path = f"checked_images/wrist_{camera_index}.png"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, frame) 
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        decoded_objects = pyzbar.decode(gray)

        current_frame_has_qr = False

        for obj in decoded_objects:
            qr_data_str = obj.data.decode('utf-8')
            current_frame_has_qr = True
            

            try:
                raw_val = int(qr_data_str)
                generator_time_ms = 0

                if raw_val < 10000000000: 
                    current_prefix = (scanner_time_ms // 1000000) * 1000000
                    
                    reconstructed_time = current_prefix + raw_val
                    
                    diff_check = scanner_time_ms - reconstructed_time
                    
                    if diff_check < -500000: 
                        reconstructed_time -= 1000000
                    elif diff_check > 500000:
                        reconstructed_time += 1000000
                        
                    generator_time_ms = reconstructed_time
                else:
                    generator_time_ms = raw_val

                time_difference = scanner_time_ms - generator_time_ms
                
                points = obj.polygon
                if len(points) > 3:
                    pts = np.array([(p.x, p.y) for p in points], dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

                # 显示文字
                text_delay = f"Delay: {time_difference} ms"
                text_ts = f"QR: {raw_val}"
                rect = obj.rect
                
                cv2.rectangle(frame, (rect.left, rect.top - 40), (rect.left + 200, rect.top), (0,0,0), -1)
                cv2.putText(frame, text_delay, (rect.left, rect.top - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                if -1000 < time_difference < 5000:
                    total_time_difference.append(time_difference)
                    print(f"延迟: {time_difference:4d} ms | 扫描器: {scanner_time_ms} | 来源: {generator_time_ms}")

            except ValueError:
                pass
        
        if len(total_time_difference) > 0:
            mean_delay = np.mean(total_time_difference[-50:])
            cv2.putText(frame, f"Avg Delay: {mean_delay:.1f} ms", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            if len(total_time_difference) >= 200:
                print("样本采集完毕，退出。")
                with open(time_difference_name, "w") as f:
                    json.dump({"average_delay_ms": np.mean(total_time_difference)}, f)
                
                break

        out.write(frame)
        # cv2.imshow("Latency Tester", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print(f"最终平均延迟: {np.mean(total_time_difference):.2f} ms")
    camera.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument("--camera_index", type=int, default=4, help="camera index for cv2.VideoCapture")
    parser.add_argument("--cap_id", type=str, default="1127.json", help="output time difference file name")
    args = parser.parse_args()
    run_qr_scanner(camera_index=args.camera_index, time_difference_name=f"calibrated_time_difference/{args.cap_id}.json", output_filename=f"calibrated_time_difference/{args.cap_id}.mp4")