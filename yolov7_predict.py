import torch
import random

from tools.lpr_predict import LPR_predict, LPR_predict_video
from UI.utils.plots import plot_one_box_PIL
from tools.operate import get_license_plate_color
from yolov7.utils.datasets import letterbox
from yolov7.utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh
from yolov7.utils.plots import plot_one_box
from tensorflow.keras.layers import Conv2D, Input, MaxPool2D, Reshape, Activation, Flatten, Dense
from tensorflow.keras.models import Model, Sequential

import cv2
import numpy as np
import math
from scipy.ndimage import gaussian_filter1d


def fitLine_ransac(pts, zero_add=0):
    '''
    该函数的作用是使用随机采用一致算法(RANSAC)对输入点进行直线拟合,并返回拟合的直线在图像中的两个端点坐标
    函数接收一个点集(pts)作为输入,使用cv2.fitline对点集进行拟合,然后根据拟合的直线斜率和截距计算出直线的
    两个端点坐标
    '''
    if len(pts) >= 2:
        [vx, vy, x, y] = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01)
        lefty = int((-x * vy / vx) + y)
        righty = int(((136 - x) * vy / vx) + y)
        return lefty + 30 + zero_add, righty + 30 + zero_add
    return 0, 0


def angle(x, y):
    '''
    计算向量(x,y)与水平方向的夹角
    '''
    return int(math.atan2(float(y), float(x)) * 180.0 / 3.1415)


def skew_detection(image_gray):
    '''
    skew_detection函数实现图像倾斜检测的功能,函数接收一张灰度图像img_gray作为输入,使用cv2.cornerEigenValsAndVecs
    函数计算图像中每个像素的特征值与特征向量,然后根据特征向量计算图像的倾斜角度
    '''
    h, w = image_gray.shape[:2]
    # 计算图像中的每个像素的特征值和特征向量将其存储到eigen向量中
    eigen = cv2.cornerEigenValsAndVecs(image_gray, 12, 5)
    # 初始化一个长度为180的数组angle_sur,所有元素设置为0(数据类型为无符号整数)
    angle_sur = np.zeros(180, np.uint)
    # 使用reshape()方法将eigen变量的形状更改为(h, w, 3, 2)
    eigen = eigen.reshape(h, w, 3, 2)
    # 提取eigen的第三个通道（索引为2）并将其赋值给flow变量
    flow = eigen[:, :, 2]
    # 创建一个与image_gray相同形状的vis变量,并将其值设置为(192 + vis的无符号整数值) / 2
    vis = image_gray.copy()
    vis[:] = (192 + np.uint32(vis)) / 2
    d = 12
    # 使用np.mgrid和reshape()方法创建一个点数组points,其形状为(-1, 2)
    points = np.dstack(np.mgrid[d / 2:w:d, d / 2:h:d]).reshape(-1, 2)
    for x, y in points:
        vx, vy = np.int32(flow[int(y), int(x)] * d)
        # cv2.line(rgb, (x-vx, y-vy), (x+vx, y+vy), (0, 355, 0), 1, cv2.LINE_AA)
        ang = angle(vx, vy)
        angle_sur[(ang + 180) % 180] += 1

    # torr_bin = 30
    angle_sur = angle_sur.astype(float)
    # 归一化处理
    angle_sur = (angle_sur - angle_sur.min()) / (angle_sur.max() - angle_sur.min())
    # 高斯滤波处理，标准差为5
    angle_sur = gaussian_filter1d(angle_sur, 5)
    # 计算垂直倾斜的最大值skew_v_val和最大值所在的索引skew_v
    skew_v_val = angle_sur[20:180 - 20].max()
    skew_v = angle_sur[30:180 - 30].argmax() + 30
    # 计算水平倾斜的两个值skew_h_A和skew_h_B,分别表示angle_sur数组在0到30之间和150到180之间的最大值
    skew_h_A = angle_sur[0:30].max()
    skew_h_B = angle_sur[150:180].max()
    skew_h = 0
    if (skew_h_A > skew_v_val * 0.3 or skew_h_B > skew_v_val * 0.3):
        if skew_h_A >= skew_h_B:
            skew_h = angle_sur[0:20].argmax()
        else:
            skew_h = - angle_sur[160:180].argmax()
    return skew_h, skew_v  # 水平和垂直方向的倾斜角度


def v_rot(img, angel, shape, max_angel):
    '''
    旋转图像并保持其内容在视觉上水平
    img:需要旋转的图像
    angel:旋转角度
    shape:输出图像的形状
    max_angel:可能的最大旋转角度
    '''
    size_o = [shape[1], shape[0]]
    size = (shape[1] + int(shape[0] * np.cos((float(max_angel) / 180) * 3.14)), shape[0])
    interval = abs(int(np.sin((float(angel) / 180) * 3.14) * shape[0]))
    # pts1的浮点数数组,表示原始图像中的四个顶点(左上、左下、右上和右下)
    pts1 = np.float32([[0, 0], [0, size_o[1]], [size_o[0], 0], [size_o[0], size_o[1]]])
    # 根据旋转角度angel的正负,定义一个名为pts2的浮点数数组,表示旋转后图像的四个顶点。
    if (angel > 0):
        pts2 = np.float32([[interval, 0], [0, size[1]], [size[0], 0], [size[0] - interval, size_o[1]]])
    else:
        pts2 = np.float32([[0, 0], [interval, size[1]], [size[0] - interval, 0], [size[0], size_o[1]]])

    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, size)
    # 返回变换后的图像dst和透视变换矩阵M
    return dst, M


def fastDeskew(image):
    '''
    fastDeskew函数用于快速矫正输入图像中的倾斜
    '''
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 获取水平倾斜角度skew_h和垂直倾斜角度skew_v
    skew_h, skew_v = skew_detection(image_gray)
    deskew, M = v_rot(image, int((90 - skew_v) * 1.5), image.shape, 60)
    return deskew, M


#精定位算法
def findContoursAndDrawBoundingBox(image_rgb,license_color="blue"):
    # 存储边界框信息
    line_upper  = []
    line_lower = []
    line_experiment = []
    grouped_rects = []
    # 将图片转化为灰度图
    gray_image = cv2.cvtColor(image_rgb,cv2.COLOR_BGR2GRAY)
    if license_color == "green":
        # 将像素值小于128的像素设置为255，否则设置为0
        ret, img_thresh = cv2.threshold(gray_image, 90, 255, cv2.THRESH_BINARY_INV)
        # 将二值图像转换为灰度图像
        gray_image = img_thresh
    # 遍历-80到0的等间隔数列,共15个元素,用于调整二值化阈值以便完成在不同亮度条件下检测轮廓
    for k in np.linspace(-80, 0, 15):
        # 对灰度图像
        binary_niblack = cv2.adaptiveThreshold(gray_image,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,17,k)
        # 在二值图像中查找轮廓,返回轮廓列表contours和层次结构hierarchy
        contours, hierarchy = cv2.findContours(binary_niblack.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            # 计算外界矩形bdbox
            bdbox = cv2.boundingRect(contour)
            # 根据外接矩形判断(例如:高宽比和面积等)是否满足某种条件
            # 如果满足将左上角坐标添加到line_upper,将其右上角坐标添加到line_lower
            # 同时将这两个点添加到line_experiment
            if (bdbox[3]/float(bdbox[2])>0.7 and bdbox[3]*bdbox[2]>100 and bdbox[3]*bdbox[2]<1200) or (bdbox[3]/float(bdbox[2])>3 and bdbox[3]*bdbox[2]<100):
                # cv2.rectangle(rgb,(bdbox[0],bdbox[1]),(bdbox[0]+bdbox[2],bdbox[1]+bdbox[3]),(255,0,0),1)
                line_upper.append([bdbox[0],bdbox[1]])
                line_lower.append([bdbox[0]+bdbox[2],bdbox[1]+bdbox[3]])

                line_experiment.append([bdbox[0],bdbox[1]])
                line_experiment.append([bdbox[0]+bdbox[2],bdbox[1]+bdbox[3]])
                # grouped_rects.append(bdbox)
    # 为输入图像添加上下边界
    rgb = cv2.copyMakeBorder(image_rgb,30,30,0,0,cv2.BORDER_REPLICATE)
    # 使用fitLine_ransac对line_lower中的点进行拟合,返回拟合直线的左端点leftyA和右端点rightyA
    leftyA, rightyA = fitLine_ransac(np.array(line_lower),3)
    rows,cols = rgb.shape[:2]
    # 返回拟合直线左端点leftyB和右端点rightyB
    leftyB, rightyB = fitLine_ransac(np.array(line_upper),-3)

    rows,cols = rgb.shape[:2]
    # 进行透视变换
    # rgb = cv2.line(rgb, (cols - 1, rightyB), (0, leftyB), (0,255, 0), 1,cv2.LINE_AA)
    pts_map1  = np.float32([[cols - 1, rightyA], [0, leftyA],[cols - 1, rightyB], [0, leftyB]])
    pts_map2 = np.float32([[136,36],[0,36],[136,0],[0,0]])
    mat = cv2.getPerspectiveTransform(pts_map1,pts_map2)
    image = cv2.warpPerspective(rgb,mat,(136,36),flags=cv2.INTER_CUBIC)

    # cv2.imshow("img", image)
    # cv2.waitKey(0)

    image,M = fastDeskew(image)

    return image

# 消除左右边界
# 定义CNN回归确定车牌图像左右边界模型
def getModel():
    input = Input(shape=[16, 66, 3])  # change this shape to [None,None,3] to enable arbitraty shape input
    x = Conv2D(10, (3, 3), strides=1, padding='valid', name='conv1')(input)
    x = Activation("relu", name='relu1')(x)
    x = MaxPool2D(pool_size=2)(x)
    x = Conv2D(16, (3, 3), strides=1, padding='valid', name='conv2')(x)
    x = Activation("relu", name='relu2')(x)
    x = Conv2D(32, (3, 3), strides=1, padding='valid', name='conv3')(x)
    x = Activation("relu", name='relu3')(x)
    x = Flatten()(x)
    output = Dense(2, name="dense")(x)
    output = Activation("relu", name='relu4')(output)
    model = Model([input], [output])
    return model


model = getModel()
model.load_weights("../weights/model12.h5")


def finemappingVertical(image):
    resized = cv2.resize(image, (66, 16))
    resized = resized.astype(np.float32) / 255
    res = model.predict(np.array([resized]))[0]
    res = res * image.shape[1]
    res = res.astype(np.uint32)
    H, T = res
    H -= 3
    if H < 0:
        H = 0
    T += 2

    if T >= image.shape[1] - 1:
        T = image.shape[1] - 1

    image = image[0:35, H:T + 2]

    image = cv2.resize(image, (int(136), int(36)))
    return image


# 预测车牌图像并返回截取的车牌图像以及在原图中框出车牌框
def yolov7_detect(img, model):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 设置图像大小
    with torch.no_grad():
        imgsz = 640
        stride = int(model.stride.max())  # model stride
        # 图像归一化
        imgsz = check_img_size(imgsz, s=stride)  # check img_size
        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        # colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
        # 加载图像
        img0 = img
        # Padded resize
        im = letterbox(img0, imgsz)[0]
        # Convert
        im = im[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        im = np.ascontiguousarray(im)
        im = torch.from_numpy(im).to(device)
        im = im.float()  # uint8 to fp16/32
        im /= 255.0  # 0 - 255 to 0.0 - 1.0
        if im.ndimension() == 3:
            im = im.unsqueeze(0)
        # Inference
        pred = model(im)[0]
        # NMS
        pred = non_max_suppression(pred)

        # 用于存放结果
        detections = []
        # Process predictions
        for i, det in enumerate(pred):  # per image
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], img0.shape).round()
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    # xyxy表示左上角和右下角的坐标
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()
                    xywh = [round(x) for x in xywh]
                    xywh = [
                        xywh[0] - xywh[2] // 2, xywh[1] - xywh[3] // 2, xywh[2],
                        xywh[3]
                    ]  # 检测到目标位置（x，y，w，h）
                    cls = names[int(cls)]  # 类别
                    conf = float(conf)  # 置信度
                    detections.append({
                        'class': cls,
                        'conf': conf,
                        'position': xywh
                    })
        if len(detections):
            # 在高度和宽度上分别增加10将车牌图像可以完全截取
            img_cut = img[xywh[1]:xywh[1] + xywh[3], xywh[0]:xywh[0] + xywh[2]]
            # 画出预测框
            img_copy = img0.copy()
            license_color = cls

            label = 'license plate' + ' ' + license_color

            plot_one_box(xyxy, img_copy, label=label, color=(0, 0, 255), line_thickness=3)
            # 对截取的蓝牌车牌图像进行校正
            image_gray = cv2.cvtColor(img_cut, cv2.COLOR_BGR2GRAY)
            skew_h, skew_v = skew_detection(image_gray)
            # print(skew_h,skew_v)
            # print(license_color)
            if abs(skew_v - 90) > 5 or abs(skew_h) > 2 and license_color == "green":
                img_cut = cv2.resize(img_cut, (136, 36))
                license_image = findContoursAndDrawBoundingBox(img_cut,license_color)
                license_image = finemappingVertical(license_image)
            elif license_color == "blue":
                img_cut = cv2.resize(img_cut, (136, 36))
                license_image = findContoursAndDrawBoundingBox(img_cut,license_color)
                license_image = finemappingVertical(license_image)
            else:
                license_image = img_cut

            license_image = cv2.resize(license_image, (240, 80))
            return license_image, img_copy,cls
        else:
            return None,img,None


# 检测车牌区域并识别车牌号码,返回时带有车牌号预测框的图片,用于视频检测
def yolov7_detect_recogntion(img, model, device, catcher, lprnet, hight_rec=False):
    with torch.no_grad():
        # 设置图像大小
        imgsz = 640
        stride = int(model.stride.max())  # model stride
        # 图像归一化
        imgsz = check_img_size(imgsz, s=stride)  # check img_size
        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names

        # 模型预先运行一次
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

        # 加载图像
        img0 = img
        # Padded resize
        im = letterbox(img0, imgsz)[0]
        # Convert
        im = im[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        im = np.ascontiguousarray(im)
        im = torch.from_numpy(im).to(device)
        im = im.float()  # uint8 to fp16/32
        im /= 255.0  # 0 - 255 to 0.0 - 1.0
        if im.ndimension() == 3:
            im = im.unsqueeze(0)
        # Inference
        pred = model(im)[0]
        # NMS
        pred = non_max_suppression(pred)

        # 用于存放结果
        detections = []
        # Process predictions
        for i, det in enumerate(pred):  # per image
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], img0.shape).round()
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()
                    xywh = [round(x) for x in xywh]
                    xywh = [
                        xywh[0] - xywh[2] // 2, xywh[1] - xywh[3] // 2, xywh[2],
                        xywh[3]
                    ]  # 检测到目标位置（x，y，w，h）
                    cls = names[int(cls)]  # 类别
                    conf = float(conf)  # 置信度
                    detections.append({
                        'class': cls,
                        'conf': conf,
                        'xyxy': xyxy,
                        'xywh': xywh
                    })

        if len(detections):
            # 画出预测框
            img_copy = img0.copy()
            if hight_rec:
                # -----------------------------高精度识别-------------------------------------
                if catcher(img0):
                    license_name = catcher(img0)[0][0]
                    label = f'{license_name}'
                    im0 = plot_one_box_PIL(xyxy, img0, label=label, color=(0, 0, 255), line_thickness=3)
                    return im0
            else:
                # -----------------------------lpr普通识别-------------------------------------
                for information in detections:
                    xyxy = information["xyxy"]
                    xywh = information["xywh"]
                    license_color = information["class"]
                    # 在高度和宽度上分别增加10将车牌图像可以完全截取
                    img_cut = img[xywh[1]:xywh[1] + xywh[3], xywh[0]:xywh[0] + xywh[2]]
                    # 对截取的蓝牌车牌图像进行校正
                    image_gray = cv2.cvtColor(img_cut, cv2.COLOR_BGR2GRAY)
                    skew_h, skew_v = skew_detection(image_gray)
                    if abs(skew_v - 90) > 5 or abs(skew_h) > 2 and license_color == "green":
                        img_cut = cv2.resize(img_cut, (136, 36))
                        license_image = findContoursAndDrawBoundingBox(img_cut, license_color)
                        license_image = finemappingVertical(license_image)
                    elif license_color == "blue":
                        img_cut = cv2.resize(img_cut, (136, 36))
                        license_image = findContoursAndDrawBoundingBox(img_cut, license_color)
                        license_image = finemappingVertical(license_image)
                    else:
                        license_image = img_cut

                    license_image = cv2.resize(license_image, (240, 80))
                    license_name = LPR_predict_video(license_image, lprnet)
                    img_copy = plot_one_box_PIL(xyxy, img_copy, label=license_name, color=(0, 0, 255), line_thickness=4)
                return img_copy
        else:
            return None



# 用于图像检测
def yolov7_detect_rec_nums(img, yolov7,lprnet):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 设置图像大小
    with torch.no_grad():
        imgsz = 640
        stride = int(yolov7.stride.max())  # model stride
        # 图像归一化
        imgsz = check_img_size(imgsz, s=stride)  # check img_size
        # Get names and colors
        names = yolov7.module.names if hasattr(yolov7, 'module') else yolov7.names
        # 加载图像
        img0 = img
        # Padded resize
        im = letterbox(img0, imgsz)[0]
        # Convert
        im = im[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        im = np.ascontiguousarray(im)
        im = torch.from_numpy(im).to(device)
        im = im.float()  # uint8 to fp16/32
        im /= 255.0  # 0 - 255 to 0.0 - 1.0
        if im.ndimension() == 3:
            im = im.unsqueeze(0)
        # Inference
        pred = yolov7(im)[0]
        # NMS
        pred = non_max_suppression(pred)

        # 用于存放结果
        detections = []
        # Process predictions
        for i, det in enumerate(pred):  # per image
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], img0.shape).round()
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()
                    xywh = [round(x) for x in xywh]
                    xywh = [
                        xywh[0] - xywh[2] // 2, xywh[1] - xywh[3] // 2, xywh[2],
                        xywh[3]
                    ]  # 检测到目标位置（x，y，w，h）
                    cls = names[int(cls)]  # 类别
                    conf = float(conf)  # 置信度
                    detections.append({
                        'class': cls,
                        'conf': conf,
                        'xyxy': xyxy,
                        'xywh':xywh
                    })
        if len(detections):
            # 画出预测框
            img_copy = img0.copy()
            for information in detections:
                xyxy = information["xyxy"]
                xywh = information["xywh"]
                license_color = information["class"]
                # 在高度和宽度上分别增加10将车牌图像可以完全截取
                img_cut = img[xywh[1]:xywh[1] + xywh[3], xywh[0]:xywh[0] + xywh[2]]
                # 对截取的蓝牌车牌图像进行校正
                image_gray = cv2.cvtColor(img_cut, cv2.COLOR_BGR2GRAY)
                skew_h, skew_v = skew_detection(image_gray)
                if abs(skew_v - 90) > 5 or abs(skew_h) > 2 and license_color == "green":
                    img_cut = cv2.resize(img_cut, (136, 36))
                    license_image = findContoursAndDrawBoundingBox(img_cut, license_color)
                    license_image = finemappingVertical(license_image)
                elif license_color == "blue":
                    img_cut = cv2.resize(img_cut, (136, 36))
                    license_image = findContoursAndDrawBoundingBox(img_cut, license_color)
                    license_image = finemappingVertical(license_image)
                else:
                    license_image = img_cut

                license_image = cv2.resize(license_image, (240, 80))
                license_name = LPR_predict(license_image, lprnet)
                img_copy = plot_one_box_PIL(xyxy, img_copy, label=license_name, color=(0,0,255), line_thickness=4)
            return img_copy
        else:
            return img0