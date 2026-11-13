import cv2
import numpy as np
from argparse import ArgumentParser

def main():
    parser = ArgumentParser(description="Preview fisheye correction on a calibration image.")
    parser.add_argument("image", help="Path to a test image")
    args = parser.parse_args()

    # 1. 读取刚才保存的标定参数
    try:
        data = np.load("fisheye_calib_result.npz")
        K = data['K']
        D = data['D']
        print("✅ 成功加载相机内参 K 和畸变系数 D")
    except Exception as e:
        print(f"❌ 无法读取 npz 文件: {e}")
        return

    # 2. 读取一张包含明显直线的测试图 (比如你发给我的那张带有门框和幕布的照片)
    # 请根据你的实际路径修改下面这张图片的名字！
    test_img_path = args.image
    img = cv2.imread(test_img_path)
    
    if img is None:
        print(f"❌ 找不到图片: {test_img_path}")
        return

    h, w = img.shape[:2]

    # 3. 计算新的相机矩阵 (这是鱼眼去畸变的关键步骤)
    # balance 参数范围在 0.0 到 1.0 之间：
    #   0.0: 裁剪掉所有黑边，只保留有效画面（视野会变小，但没有黑边）
    #   1.0: 保留所有原始像素（视野最大，但周围会有严重的拉伸变形和黑边）
    # 我们设为 0.5 作为折中，你可以自己修改试试看效果
    balance = 0.5
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, (w, h), np.eye(3), balance=balance
    )

    # 4. 生成映射表并执行重映射 (去畸变)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), new_K, (w, h), cv2.CV_16SC2
    )
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    # 5. 显示对比结果 (为了适应屏幕，稍微缩小显示)
    display_scale = 0.5 if w > 1000 else 1.0
    
    cv2.imshow("Original (Distorted)", cv2.resize(img, (0, 0), fx=display_scale, fy=display_scale))
    cv2.imshow(f"Undistorted (Balance={balance})", cv2.resize(undistorted_img, (0, 0), fx=display_scale, fy=display_scale))

    print(f"👀 请观察两张图片的对比。按任意键关闭窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # 保存去畸变后的图片以便仔细查看
    save_path = "undistorted_result.jpg"
    cv2.imwrite(save_path, undistorted_img)
    print(f"💾 去畸变后的图片已保存为: {save_path}")

if __name__ == "__main__":
    main()