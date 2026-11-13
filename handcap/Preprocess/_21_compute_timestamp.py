import os
import cv2
import time
import json
import numpy as np
from pyzbar import pyzbar
from argparse import ArgumentParser

def run_qr_scanner(camera_index=4, time_difference_name="1127", output_filename="1127.mp4"):
    os.makedirs("calibrated_time_difference", exist_ok=True)
    camera = cv2.VideoCapture(camera_index)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1) 
    
    # 尝试设置高分辨率以匹配你的内参 (960*2, 540*2)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    if not camera.isOpened():
        print(f"错误：无法打开摄像头索引 {camera_index}。")
        return

    frame_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 30.0 

    # --- 新增: 鱼眼矫正设置 (关键步骤) ---
    # 1. 定义你的内参矩阵 K
    K = np.array([[480.,   0., 960.],
                  [  0., 480., 540.],
                  [  0.,   0.,   1.]])
    
    # 2. 定义畸变系数 D (你需要填入你标定出的真实值!)
    # 如果没有标定，这里是一个典型的鱼眼大概率值，需要微调
    # 格式通常为 [k1, k2, k3, k4]
    D = np.array([0, 0, 0, 0], dtype=np.float64) 

    # 3. 计算新的相机矩阵 (保留更多图像细节，调整 balance)
    # balance=0: 裁剪掉黑色边缘 (保留中心)
    # balance=1: 保留所有像素 (图像周围会有黑色弯曲边界)
    dim = (frame_width, frame_height)
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, dim, np.eye(3), balance=0.5)
    
    # 4. 预计算映射表 (提速关键: 不要放在 while 循环里)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, dim, cv2.CV_16SC2)
    # ------------------------------------

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))
    
    print(f"摄像头已启动 (分辨率: {frame_width}x{frame_height})")
    print("已启用鱼眼畸变校正。")
    
    total_time_difference = []
    frame_count = 0 
    os.makedirs("checked_images", exist_ok=True)
    
    while True:
        scanner_now = time.time()
        scanner_time_ms = int(scanner_now * 1000)
        
        ret, frame = camera.read()
        if not ret:
            break
            
        frame_count += 1
        
        # --- 新增: 应用畸变校正 ---
        # 这一步非常快，因为 map 已经计算好了
        undistorted_frame = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        # ------------------------

        if frame_count % 20 == 0:
            output_path = f"checked_images/wrist_{camera_index}.png"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, frame) 
        
        # 使用校正后的图像进行检测
        # 注意：为了提高速度，可以只对灰度图做 remap，或者在这里转灰度
        gray = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2GRAY)
        
        # 可选优化: 自适应二值化 (对抗光照不均)
        # gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 2)

        decoded_objects = pyzbar.decode(gray)

        # 绘制逻辑需要在 undistorted_frame 上进行
        display_frame = undistorted_frame.copy() 

        for obj in decoded_objects:
            qr_data_str = obj.data.decode('utf-8')
            
            try:
                raw_val = int(qr_data_str)
                # 简化逻辑用于演示
                if raw_val < 10000000000:
                    current_prefix = (scanner_time_ms // 1000000) * 1000000
                    generator_time_ms = current_prefix + raw_val
                else:
                    generator_time_ms = raw_val

                time_difference = scanner_time_ms - generator_time_ms
                
                # 绘制
                points = obj.polygon
                if len(points) > 3:
                    pts = np.array([(p.x, p.y) for p in points], dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(display_frame, [pts], True, (0, 255, 0), 2)

                text_delay = f"Delay: {time_difference} ms"
                rect = obj.rect
                cv2.rectangle(display_frame, (rect.left, rect.top - 40), (rect.left + 200, rect.top), (0,0,0), -1)
                cv2.putText(display_frame, text_delay, (rect.left, rect.top - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                if -1000 < time_difference < 5000:
                    total_time_difference.append(time_difference)
                    print(f"延迟: {time_difference:4d} ms")

            except ValueError:
                pass
        
        if len(total_time_difference) > 0:
            mean_delay = np.mean(total_time_difference[-50:])
            cv2.putText(display_frame, f"Avg Delay: {mean_delay:.1f} ms", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            if len(total_time_difference) >= 200:
                print("样本采集完毕...")
                with open(time_difference_name, "w") as f:
                    json.dump({"average_delay_ms": np.mean(total_time_difference)}, f)
                break

        # 保存/显示的是校正后的图
        out.write(display_frame) 
        # cv2.imshow("Undistorted View", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print(f"最终平均延迟: {np.mean(total_time_difference):.2f} ms")
    camera.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--camera_index", type=int, default=4) # 注意这里改回了0，根据实际情况调整
    parser.add_argument("--cap_id", type=str, default="1127.json")
    args = parser.parse_args()
    run_qr_scanner(camera_index=args.camera_index, time_difference_name=f"calibrated_time_difference/{args.cap_id}.json", output_filename=f"calibrated_time_difference/{args.cap_id}.mp4")