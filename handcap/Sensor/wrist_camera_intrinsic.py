
import os
import cv2
import numpy as np
import glob

CHECKERBOARD = (6, 9) # 棋盘格内角点数量 (行, 列)，根据你的棋盘格修改
SQUARE_SIZE = 20 # 棋盘格方块大小 (mm)
subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW

objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * SQUARE_SIZE

camera_index = 4

cap = cv2.VideoCapture(camera_index) # 修改为对应的相机索引
# 设置相机初始参数
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 25)

objpoints = []
imgpoints = []

found = 0

os.makedirs('output_corners', exist_ok=True)

while True:
    ret_val, img = cap.read()
    if not ret_val:
        print("无法读取相机帧")
        break
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FILTER_QUADS
    # ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, flags)
    ret, corners = cv2.findChessboardCornersSB(gray, CHECKERBOARD, cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY)
    
    cv2.imwrite(f'output_corners/corners.jpg', cv2.drawChessboardCorners(img, CHECKERBOARD, corners, ret))
    print(ret)
    if ret == True:
        found += 1
        objpoints.append(objp)
        cv2.cornerSubPix(gray, corners, (3,3), (-1,-1), subpix_criteria)
        imgpoints.append(corners)
        print(f"找到 {found} 张有效图片...")
        
        if found >= 20: 
            break

print(f"找到 {found} 张有效图片，开始计算矩阵...")

# 3. 核心标定函数 (注意使用的是 cv2.fisheye)
N_OK = len(objpoints)
K = np.zeros((3, 3))
D = np.zeros((4, 1))
rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]

rms, _, _, _, _ = cv2.fisheye.calibrate(
    objpoints, imgpoints, gray.shape[::-1],
    K, D, rvecs, tvecs, calibration_flags,
    (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
)

print(f"标定完成！RMS误差: {rms}")
print("内参矩阵 K:\n", K)
print("畸变系数 D:\n", D)

# 保存参数以便后续使用
np.savez("fisheye_params.npz", K=K, D=D)