import cv2
import numpy as np
import glob
import os

# ==========================================
# 🛑 第一步：核心参数配置（运行前必须修改！）
# ==========================================

# 1. 棋盘格的内角点数 (列数, 行数) 
# 注意：不是黑白方块的数量！是内部十字交叉点的数量！
# 比如 9x6 的方块，内角点就是 (8, 5)
CHECKERBOARD = (6, 4) 

# 2. 每个方格的实际物理边长 (单位可以是毫米 mm 或米 m，保持统一即可)
# 强烈建议用游标卡尺测量一下，比如 20.0 毫米
SQUARE_SIZE = 37.9 #34  

# 3. 图像所在的文件夹路径
IMAGE_DIR = os.environ.get("HANDCAP_CALIBRATION_IMAGES", "_calib_images")
IMAGE_FORMAT = "*.jpg"

# ==========================================

def main():
    print("🚀 开始鱼眼相机标定...")
    
    # 亚像素角点提取的终止条件 (最大迭代30次，或者精度达到0.1)
    subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    
    # 鱼眼标定的标志位 (固定倾斜度、检查条件等，提高鲁棒性)
    #calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_CHECK_COND | cv2.fisheye.CALIB_FIX_SKEW
    # 移除 CALIB_CHECK_COND，允许算法在稍微病态的矩阵下继续尝试收敛
    calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW

    # 准备 3D 真实世界坐标点
    # 鱼眼标定要求的维度比较严格：(1, N, 3) 且类型为 float32
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * SQUARE_SIZE

    # 用于存储所有图像的 3D 点和 2D 像素点
    objpoints = [] # 真实世界中的 3D 点
    imgpoints = [] # 图像平面中的 2D 像素点

    # 获取所有图片路径
    images = glob.glob(os.path.join(IMAGE_DIR, IMAGE_FORMAT))
    if not images:
        print(f"❌ 错误：在 {IMAGE_DIR} 目录下没有找到 {IMAGE_FORMAT} 图片！")
        return

    print(f"📁 找到 {len(images)} 张图片，开始提取角点...")
    
    img_shape = None
    success_count = 0

    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        if img_shape is None:
            img_shape = img.shape[:2] # (height, width)
        else:
            assert img_shape == img.shape[:2], "❌ 所有图片的尺寸必须完全一致！"

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 寻找棋盘格角点
        ret, corners = cv2.findChessboardCorners(
            gray, 
            CHECKERBOARD, 
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if ret:
            # 找到角点后，在原图像基础上寻找亚像素级别的精确角点
            corners_subpix = cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), subpix_criteria)
            
            # 强制转换为 (N, 1, 3) 和 (N, 1, 2) 的维度
            objpoints.append(objp.reshape(-1, 1, 3))
            imgpoints.append(corners_subpix.reshape(-1, 1, 2))


            success_count += 1
            print(f"  ✅ [{idx+1}/{len(images)}] 角点提取成功: {os.path.basename(fname)}")
            
            # 可选：绘制并显示角点（如果想看的话把下面两行取消注释）
            # cv2.drawChessboardCorners(img, CHECKERBOARD, corners_subpix, ret)
            # cv2.imshow('Corners', cv2.resize(img, (800, 800)))
            # cv2.waitKey(100)
        else:
            print(f"  ❌ [{idx+1}/{len(images)}] 无法检测到完整角点: {os.path.basename(fname)}")

    cv2.destroyAllWindows()
    print(f"\n✨ 角点提取完成。成功图片数: {success_count}/{len(images)}")
    
    if success_count < 10:
        print("⚠️ 警告：成功提取角点的图片太少（少于10张），标定结果可能不准确，建议重新拍摄。")

    print("\n🧠 正在进行非线性优化解算 (可能需要几秒钟)...")
    
    # 初始化相机内参矩阵 K 和 畸变系数 D
    # ================= 核心修改区 =================
    K = np.zeros((3, 3), np.float64)
    D = np.zeros((4, 1), np.float64)
    
    # 动态获取图像的高和宽
    h, w = gray.shape[:2]
    
    # 1. 给出符合物理常识的初始猜测值
    # 鱼眼镜头的焦距通常在图像宽度的 1/3 到 1/4 左右
    focal_guess = w / 3.0 
    K[0, 0] = focal_guess      # fx
    K[1, 1] = focal_guess      # fy
    K[0, 2] = w / 2.0          # cx (强制放在图像绝对水平中心)
    K[1, 2] = h / 2.0          # cy (强制放在图像绝对垂直中心)
    K[2, 2] = 1.0
    
    # 2. 更新标志位：强制使用我们的猜测值，并且锁死光心不让算法乱改
    calibration_flags = (
        cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | 
        cv2.fisheye.CALIB_FIX_SKEW | 
        cv2.fisheye.CALIB_USE_INTRINSIC_GUESS    # 启用初始猜测
        #cv2.fisheye.CALIB_FIX_PRINCIPAL_POINT     # 锁死画面中心
    )
    # ==============================================

    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(success_count)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(success_count)]

    # 核心：执行鱼眼标定
    rms, _, _, _, _ = cv2.fisheye.calibrate(
        objpoints,
        imgpoints,
        gray.shape[::-1],
        K,
        D,
        rvecs,
        tvecs,
        calibration_flags,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6) # 增加最大迭代次数到100
    )

    print("\n==========================================")
    print("🏆 标定结果")
    print("==========================================")
    print(f"📌 RMS 重投影误差: {rms:.4f} 像素")
    if rms < 0.5:
        print("  -> 误差极小，标定质量非常优秀！")
    elif rms < 1.0:
        print("  -> 误差尚可，标定质量一般。")
    else:
        print("  -> 误差偏大，建议检查角点提取是否准确，或者重新拍摄！")

    print("\n📷 相机内参矩阵 K (Camera Matrix):")
    print(K)
    print(f"   fx (X轴焦距) = {K[0,0]:.2f}")
    print(f"   fy (Y轴焦距) = {K[1,1]:.2f}")
    print(f"   cx (X轴光心) = {K[0,2]:.2f}")
    print(f"   cy (Y轴光心) = {K[1,2]:.2f}")

    print("\n🌀 鱼眼畸变系数 D (Distortion Coefficients k1, k2, k3, k4):")
    print(D.T)
    print("==========================================")

    # 保存结果到 NumPy 文件，方便后续读取
    np.savez("fisheye_calib_result.npz", K=K, D=D, rms=rms)
    print("💾 标定参数已保存为 fisheye_calib_result.npz")

if __name__ == "__main__":
    main()