import cv2
import numpy as np
import glob
import os

# ================= 配置区域 =================
# 1. 棋盘格参数
CHECKERBOARD = (6, 9)  # 棋盘格内角点数量 (行, 列) -> 数黑白格交界点，不是格子数
SQUARE_SIZE = 20       # 每个格子的实际边长 (单位: mm)，这影响平移向量的单位，不影响畸变系数

# 2. 图片路径
# IMAGE_PATH = './calibration_images/*.jpg'  # 修改为您图片的实际扩展名

# 3. 硬件参数 (用于计算初始猜测) - 基于您提供的 HF890 + 1.44mm 镜头
IMG_WIDTH = 1920       # [cite: 1]
IMG_HEIGHT = 1080      # [cite: 1]
SENSOR_PIXEL_SIZE = 0.003  # 3um [cite: 1]
FOCAL_LENGTH_MM = 1.44     # 

# ===========================================

def run_calibration():
    # 1. 准备对象点 (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp = objp * SQUARE_SIZE

    camera_index = 4

    cap = cv2.VideoCapture(camera_index) # 修改为对应的相机索引
    # 设置相机初始参数
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    found_count = 0
    os.makedirs('output_corners', exist_ok=True)
    while True:
        ret_val, img = cap.read()
        if not ret_val:
            print("无法读取相机帧")
            break
        

        # 强制转换为灰度
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 寻找棋盘格角点
        # 鱼眼镜头边缘畸变大，可能需要调整 flags
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, flags)

        cv2.imwrite(f'output_corners/corners_{found_count}.jpg', cv2.drawChessboardCorners(img, CHECKERBOARD, corners, ret))
        
        if ret:
            objpoints.append(objp)
            
            # 亚像素级角点精确化
            corners2 = cv2.cornerSubPix(gray, corners, (3,3), (-1,-1), 
                                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))
            imgpoints.append(corners2)
            found_count += 1
            print(f"[OK] {found_count}: 角点已找到")
        else:
            print(f"[FAIL] {found_count} - 未找到角点")

    print(f"成功在 {found_count} 张图片中找到角点。")
    
    if found_count < 5:
        print("有效图片太少，建议至少拍摄 10 张以上包含不同角度的图片。")
        return

    # 2. 设置初始内参猜测 (Initial Guess)
    # 基于之前计算的理论值，这对于大视场鱼眼非常重要，防止优化跑偏
    # f_pixels = FOCAL_LENGTH_MM / SENSOR_PIXEL_SIZE # 480
    # cx = IMG_WIDTH / 2
    # cy = IMG_HEIGHT / 2
    
    K_guess = np.array([[480.,   0., 960.],
    [  0., 480., 540.],
    [  0.,   0.,   1.]
    ], dtype=np.float64)
    
    D_guess = np.zeros((4, 1), dtype=np.float64)

    # 3. 执行鱼眼标定
    # 关键 Flags: 
    # CALIB_USE_INTRINSIC_GUESS: 使用我们计算的 K_guess 作为起点
    # CALIB_RECOMPUTE_EXTRINSIC: 对鱼眼模型很重要
    # CALIB_FIX_SKEW: 也就是 K[0,1] 固定为 0
    calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + \
                        cv2.fisheye.CALIB_CHECK_COND + \
                        cv2.fisheye.CALIB_FIX_SKEW + \
                        cv2.fisheye.CALIB_USE_INTRINSIC_GUESS

    print("\n开始计算参数 (可能需要几秒钟)...")
    try:
        rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
            objpoints, 
            imgpoints, 
            gray.shape[::-1], 
            K_guess, 
            D_guess, 
            flags=calibration_flags
        )
    except cv2.error as e:
        print("标定失败，通常是因为某些图像的角点顺序错误或图像质量太差。")
        print(e)
        return

    # 4. 输出结果
    print("\n" + "="*40)
    print(f"标定完成! RMS 误差: {rms:.4f} (越小越好，通常 < 0.5 优秀，< 1.0 可用)")
    print("="*40)
    print("内参矩阵 K (Intrinsics):")
    print(np.round(K, 3))
    print("\n畸变系数 D (Distortion [k1, k2, k3, k4]):")
    print(np.round(D, 4))
    
    # 5. 保存结果
    np.savez("fisheye_params.npz", K=K, D=D)
    print("\n参数已保存至 'fisheye_params.npz'")

    # 6. 测试去畸变效果 (Undistort Test)
    print("正在生成测试去畸变图像...")
    test_img = cv2.imread(images[0])
    h, w = test_img.shape[:2]
    
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (w,h), np.eye(3), balance=1)
    
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, (w,h), cv2.CV_16SC2)
    undistorted_img = cv2.remap(test_img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    
    cv2.imwrite("undistorted_result.jpg", undistorted_img)
    print("去畸变测试图已保存为 'undistorted_result.jpg'")

if __name__ == "__main__":
    run_calibration()