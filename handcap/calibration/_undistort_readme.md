Python 读取示例

可以使用以下代码片段在任何 Python 脚本中加载参数：

```python
import numpy as np
import cv2

# 加载标定结果
def load_calibration(file_path='fisheye_calib_result.npz'):
    with np.load(file_path) as data:
        K = data['K']
        D = data['D']
        rms = data['rms']
    return K, D, rms

# 快速应用：去畸变
def undistort_frame(frame, K, D, balance=0.0):
    h, w = frame.shape[:2]
    # 计算新的相机矩阵以获得更好的显示效果
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, (w, h), np.eye(3), balance=balance
    )
    # 生成映射表
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), new_K, (w, h), cv2.CV_16SC2
    )
    # 执行重映射
    undistorted_img = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)
    return undistorted_img

# 使用示例
if __name__ == "__main__":
    K, D, rms = load_calibration()
    print(f"成功加载内参，RMS误差: {rms:.4f}")
    
    # 假设有一帧图像 frame
    # frame = cv2.imread('test.jpg')
    # result = undistort_frame(frame, K, D)