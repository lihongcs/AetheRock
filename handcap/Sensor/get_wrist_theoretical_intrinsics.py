import numpy as np
import cv2

# --- 1. 根据规格书硬编码的参数 ---
EFL_MM = 1.44           # 物理焦距 
PIXEL_SIZE_MM = 0.003   # 像元大小 3um 
IMG_W = 1920            # 分辨率宽 
IMG_H = 1080            # 分辨率高 

# --- 2. 计算理论内参 K ---
f_pixels = EFL_MM / PIXEL_SIZE_MM
cx = IMG_W / 2
cy = IMG_H / 2

K_theoretical = np.array([
    [f_pixels, 0, cx],
    [0, f_pixels, cy],
    [0, 0, 1]
], dtype=np.float32)

print("理论内参矩阵 K (基于出厂参数):")
print(K_theoretical)

# --- 3. 关于畸变系数 D ---
# 警告：文档中的 "-99.57%" 无法直接转换为 k1, k2, k3, k4
# 对于鱼眼镜头，初始猜测通常设为 0，让标定算法去拟合
D_theoretical = np.zeros((4, 1), dtype=np.float32) 

print("\n畸变系数 D (建议通过标定获取):")
print(D_theoretical.T)

# --- 4. 标定建议 ---
print("-" * 30)
print("下一步建议：")
print("由于是 180度 鱼眼镜头，请务必使用 cv2.fisheye 模块进行标定。")
print(f"标定时，请将上述 K_theoretical 作为 flags=cv2.fisheye.CALIB_USE_INTRINSIC_GUESS 的输入。")
