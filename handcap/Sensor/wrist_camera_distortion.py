import cv2
import numpy as np

# 加载参数 (或者手动填入上一步算出的数值)
# data = np.load("fisheye_params.npz")
# K = data['K']
# D = data['D']

# 示例：手动填入（如果没有运行第一步，可以用这些经验值尝试，但建议实测）
# 假设图片宽1920，高1080
W, H = 1920, 1080
K = np.array([[W/3, 0, W/2], [0, W/3, H/2], [0, 0, 1]], dtype=np.float32) 
D = np.array([[-0.1], [0.01], [-0.01], [0.001]], dtype=np.float32) # 鱼眼通常不需要像普通镜头那么多的畸变系数，主要是K1-K4

img = cv2.imread('fisheye_sample.jpg')
h, w = img.shape[:2]

# --- 核心去畸变逻辑 ---

# 调整新的相机矩阵 (balance参数控制视场: 0=裁掉黑边但丢失视野, 1=保留所有视野但有黑边)
balance = 1.0 
new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (w, h), np.eye(3), balance=balance)

# 生成映射表
map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, (w, h), cv2.CV_16SC2)

# 重映射
undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

cv2.imshow("Original", img)
cv2.imshow("Undistorted", undistorted_img)
cv2.waitKey(0)
cv2.destroyAllWindows()