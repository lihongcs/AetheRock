import cv2
import os
# 注意：这里改为了从 mvs_cam_new 导入
from Sensor.mvs_cam import MVSCamera

def main():
    # 1. 设置并创建保存路径
    save_dir = os.environ.get("HANDCAP_CALIBRATION_IMAGES", "_calib_images")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"📁 已创建保存目录: {save_dir}")

    # 2. 初始化相机配置
    # ⚠️ 注意：请确保 serial 是你当前连接的相机序列号
    cam_config = {
        "serial": os.environ.get("HANDCAP_WRIST_CAMERA_SERIAL"),
        "exposure": 20000       
    }
    
    print("正在初始化相机...")
    try:
        cam = MVSCamera(cam_config)
        print("✅ 相机初始化成功！(自动增益已关闭，当前固定增益为 18.0)")
    except Exception as e:
        print(f"❌ 相机初始化失败: {e}")
        return

    img_count = 0

    print("\n=======================================================")
    print(" 📷 标定图像采集工具已启动")
    print(" 操作说明:")
    print(" 1. 选中弹出的实时画面窗口")
    print(" 2. 按 's' 键保存当前图像 (建议拍摄 15-25 张)")
    print(" 3. 按 'q' 或 'ESC' 键退出程序")
    print(" 💡 提示：请确保拍摄时改变标定板的位置和倾斜角度！")
    print("=======================================================\n")

    # 3. 循环读取并显示画面
    while True:
        # 读取一帧画面
        ret, frame, timestamp = cam.read()
        
        if not ret:
            print("⚠️ 读取画面失败，正在重试...")
            cv2.waitKey(10)
            continue

        # 拷贝一帧用于UI显示，避免污染要保存的原始图像
        display_frame = frame.copy()
        
        # 在画面左上角添加 UI 提示
        cv2.putText(display_frame, f"Saved count: {img_count}/25", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(display_frame, "[S]ave  |  [Q]uit", (20, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

        # 智能等比例缩小显示窗口（限制最大边长为 800 像素，适应所有屏幕）
        # 这里仅作预览缩放，绝对不影响最终按 's' 键保存的高分辨率原始图像！
        h, w = display_frame.shape[:2]
        max_preview_size = 800  # 如果你觉得窗口还是太大或太小，可以修改这个数字
        
        if h > max_preview_size or w > max_preview_size:
            scale = min(max_preview_size / w, max_preview_size / h)
            new_w, new_h = int(w * scale), int(h * scale)
            preview = cv2.resize(display_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            preview = display_frame

        cv2.imshow("Calibration Capture Viewer", preview)

        # 监听键盘事件，延迟1毫秒
        key = cv2.waitKey(1) & 0xFF

        # 按 's' 键保存图片
        if key == ord('s'):
            # 统一命名格式，例如 calib_00.jpg, calib_01.jpg
            img_name = os.path.join(save_dir, f"calib_{img_count:02d}.jpg")
            
            # ⚠️ 关键：保存的是原始的高分辨率 frame，而不是带文字的 display_frame
            cv2.imwrite(img_name, frame)
            print(f"📸 成功保存: {img_name}")
            img_count += 1

        # 按 'q' 或 ESC (ASCII码 27) 键退出
        elif key == ord('q') or key == 27:
            print(f"\n⏹ 采集结束，共保存了 {img_count} 张图像。")
            break

    # 4. 释放相机和窗口资源
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()