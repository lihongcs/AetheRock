import cv2
import os
import numpy as np
# 导入你的相机驱动模块
from Sensor.mvs_cam import MVSCamera

def load_calibration(file_path='fisheye_calib_result.npz'):
    """读取标定好的原始内参和畸变系数"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到标定文件: {file_path}。请先完成标定并生成该文件。")
    
    with np.load(file_path) as data:
        K = data['K']
        D = data['D']
        rms = data['rms']
    return K, D, rms

def setup_fisheye_undistort(K_raw, D_raw, original_resolution, target_resolution=(640, 480), balance=0.0):
    """计算目标分辨率下的新内参 K_new，并生成重映射表 map_x, map_y"""
    K_new = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K_raw, D_raw, original_resolution, np.eye(3), 
        balance=balance, new_size=target_resolution, fov_scale=1.0
    )
    
    # 生成 CV_16SC2 格式的映射表，极大加快实时 remap 的速度
    map_x, map_y = cv2.fisheye.initUndistortRectifyMap(
        K_raw, D_raw, np.eye(3), K_new, target_resolution, cv2.CV_16SC2
    )
    
    return K_new, map_x, map_y

def main():
    # 1. 设置保存路径 (用于保存处理后的 640x480 图像进行测试)
    save_dir = os.environ.get("HANDCAP_UNDISTORTED_IMAGES", "_processed_images")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"📁 已创建保存目录: {save_dir}")

    # 2. 加载原始标定文件
    print("正在加载相机标定参数...")
    try:
        # 请确保该 npz 文件路径正确
        K_raw, D_raw, rms = load_calibration('fisheye_calib_result.npz')  # 这里加载的是原始的 K 和 D，未经过任何调整
        print(f"✅ 标定参数加载成功 (RMS 误差: {rms:.4f})")
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return

    # 3. 初始化相机配置
    cam_config = {
        "serial": os.environ.get("HANDCAP_WRIST_CAMERA_SERIAL"),
        "exposure": 20000       
    }
    
    print("正在初始化相机...")
    try:
        cam = MVSCamera(cam_config)
        print("✅ 相机初始化成功！")
    except Exception as e:
        print(f"❌ 相机初始化失败: {e}")
        return

    # 4. 抓取第一帧以获取原始分辨率，并生成映射表
    print("正在抓取首帧并初始化图像映射表...")
    ret, first_frame, timestamp = cam.read()
    if not ret:
        print("❌ 无法读取相机首帧画面，程序退出。")
        cam.release()
        return
        
    orig_h, orig_w = first_frame.shape[:2]
    ORIGINAL_RESOLUTION = (orig_w, orig_h)
    TARGET_RESOLUTION = (640, 480)
    print(f"📷 自动检测到相机原始分辨率: {ORIGINAL_RESOLUTION}")

    # 计算映射表，仅在此处执行一次！
    K_new, map_x, map_y = setup_fisheye_undistort(
        K_raw, D_raw, ORIGINAL_RESOLUTION, TARGET_RESOLUTION, balance=0.0
    )
    print("\n--- 【重要】请将以下新内参输入给深度估计模型 ---")
    print(K_new)
    print("----------------------------------------------------\n")

    img_count = 0

    print("=======================================================")
    print(" 🚀 实时去畸变与Resize控制台已启动")
    print(f" 输出分辨率: {TARGET_RESOLUTION}")
    print(" 操作说明:")
    print(" 1. 选中弹出的实时画面窗口")
    print(" 2. 按 's' 键保存当前已处理的图像 (640x480)")
    print(" 3. 按 'q' 或 'ESC' 键退出程序")
    print("=======================================================\n")

    # 5. 实时控制主循环
    while True:
        ret, frame, timestamp = cam.read()
        if not ret:
            print("⚠️ 读取画面失败，正在重试...")
            cv2.waitKey(10)
            continue

        # 核心操作：查表去畸变并自动改变尺寸为 640x480
        # 这里processed_frame 经 map_x 和 map_y 映射得到，可视为一畸变系数 D=0 ，相机内参为 K_new，分辨率为640*480 的理想针孔相机的输出
        processed_frame = cv2.remap(
            frame, map_x, map_y, 
            interpolation=cv2.INTER_LINEAR, 
            borderMode=cv2.BORDER_CONSTANT
        )

        # UI 显示：直接拷贝处理后的 640x480 图像添加文字
        display_frame = processed_frame.copy()
        cv2.putText(display_frame, f"Saved: {img_count}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, "[S]ave | [Q]uit", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)

        # 640x480 尺寸已经很小，可以直接1:1显示，无需使用等比例缩小逻辑
        cv2.imshow("Processed Viewer (640x480)", display_frame)

        key = cv2.waitKey(1) & 0xFF

        # 按 's' 键保存实际传递给深度模型的处理后图片
        if key == ord('s'):
            img_name = os.path.join(save_dir, f"processed_{img_count:04d}.jpg")
            # 保存不含文字的干净图像
            cv2.imwrite(img_name, processed_frame)
            print(f"📸 成功保存去畸变后的图像: {img_name}")
            img_count += 1

        elif key == ord('q') or key == 27:
            print(f"\n⏹ 程序结束，共保存了 {img_count} 张处理后的图像。")
            break

    # 6. 释放资源
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()