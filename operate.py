import cv2
import numpy as np


# 读取图像名称中含有中文字符的图像
def imread_with_chinese_path(file_path):
    with open(file_path, 'rb') as f:
        image_data = f.read()
        image_data = np.asarray(bytearray(image_data), dtype="uint8")
        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    return image





# 计算车牌颜色
def get_license_plate_color(license_plate_image):
    # 将车牌图像从 BGR 转换为 HSV
    hsv_image = cv2.cvtColor(license_plate_image, cv2.COLOR_BGR2HSV)

    # 定义可能的车牌颜色范围
    color_ranges = {
        "blue": ([100, 40, 40], [130, 255, 255]),
        "green": ([50, 100, 100], [70, 255, 255]),
        # 添加其他颜色范围
    }

    max_color_pixels = 0
    detected_color = None

    for color, (lower, upper) in color_ranges.items():
        # 创建掩模，保留该颜色范围的像素
        mask = cv2.inRange(hsv_image, np.array(lower), np.array(upper))

        # 计算该颜色范围的像素数量
        color_pixels = cv2.countNonZero(mask)

        # 如果当前颜色范围的像素数量大于之前的最大值，更新最大值和检测到的颜色
        if color_pixels > max_color_pixels:
            max_color_pixels = color_pixels
            detected_color = color

    return detected_color


