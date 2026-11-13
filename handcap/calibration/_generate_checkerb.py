import cv2
import numpy as np
import os

def generate_chessboard(
    cols, 
    rows, 
    square_size_px=100, 
    border_size_px=50, 
    save_path="generated_chessboard.jpg"
):
    """
    生成一张标准的黑白棋盘格图片并保存。

    参数:
        cols (int): 棋盘格的内部角点列数 (比实际方块列数少 1)
        rows (int): 棋盘格的内部角点行数 (比实际方块行数少 1)
        square_size_px (int): 每个方块在图像中的像素边长
        border_size_px (int): 棋盘格周围白色边框的宽度（像素）
        save_path (str): 生成图片的保存路径
    """
    # 物理方块的数量等于内角点数 + 1
    num_squares_x = cols + 1
    num_squares_y = rows + 1

    # 1. 创建一个全黑的画布
    board_width = num_squares_x * square_size_px
    board_height = num_squares_y * square_size_px
    
    # 画布数据类型必须是 uint8 (0-255)，0 代表黑色
    board = np.zeros((board_height, board_width), dtype=np.uint8)

    # 2. 绘制白色方块形成棋盘格
    for y in range(num_squares_y):
        for x in range(num_squares_x):
            # 交替填充白色 (255)
            # 如果行索引和列索引的奇偶性相同，就是白色方块
            if (x + y) % 2 == 0:
                start_x = x * square_size_px
                end_x = start_x + square_size_px
                start_y = y * square_size_px
                end_y = start_y + square_size_px
                
                board[start_y:end_y, start_x:end_x] = 255

    # 3. 添加一个白色的外边框 (为了在打印/显示时更容易和背景区分)
    # copyMakeBorder 可以很方便地在图像四周加边框
    board_with_border = cv2.copyMakeBorder(
        board, 
        border_size_px, border_size_px, border_size_px, border_size_px, 
        cv2.BORDER_CONSTANT, 
        value=[255, 255, 255] # 边框颜色为白色
    )

    # 4. 保存图片
    cv2.imwrite(save_path, board_with_border)
    print(f"✅ 成功生成棋盘格并保存至: {os.path.abspath(save_path)}")
    print(f"   - 内部角点数: {cols} x {rows}")
    print(f"   - 图像总分辨率: {board_with_border.shape[1]} x {board_with_border.shape[0]}")


if __name__ == "__main__":
    # ================= 配置区 =================
    
    # 你的内角点数 (跟标定时代码里的 CHECKERBOARD 一致)
    # 比如你要生成一张有 6x4 个内角点的标定板
    TARGET_COLS = 6
    TARGET_ROWS = 4
    
    # 每个方块占据的像素 (你可以调大一点让生成的图片分辨率更高)
    PIXELS_PER_SQUARE = 400 
    
    # 外围留白的像素宽度
    BORDER_PIXELS = 100 
    
    # 保存文件名
    OUTPUT_FILE = "my_custom_chessboard.png"
    
    # =========================================

    generate_chessboard(
        cols=TARGET_COLS, 
        rows=TARGET_ROWS, 
        square_size_px=PIXELS_PER_SQUARE, 
        border_size_px=BORDER_PIXELS,
        save_path=OUTPUT_FILE
    )