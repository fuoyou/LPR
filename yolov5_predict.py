import numpy as np
import torch

from yolov5.utils.augmentations import letterbox
from yolov5.utils.general import (check_img_size, cv2, non_max_suppression,
                                  scale_coords, xyxy2xywh)
from yolov5.utils.plots import Annotator, colors
from yolov5.utils.torch_utils import select_device


def yolov5_detect(img, model):
    imgsz = (640, 640)
    stride, names = model.stride, model.names
    # print(stride,names)
    imgsz = check_img_size(imgsz, s=stride)
    model.model.float()
    # Run inference
    model.warmup(imgsz=(1, 3, *imgsz))  # warmup

    im0 = img
    # Padded resize
    im = letterbox(im0, new_shape=imgsz)[0]

    # Convert
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)

    im = torch.from_numpy(im)
    im = im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim

    # Inference
    pred = model(im,visualize=False)[0]
    # NMS
    pred = non_max_suppression(pred)

    # 用于存放结果
    detections = []
    # Process predictions
    for i, det in enumerate(pred):  # per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4],im0.shape).round()

            # Write results
            for *xyxy, conf, cls in reversed(det):
                # xyxy是一个长度为4的列表,表示目标框的左上角坐标和右下角坐标
                # c1,c2 = (int(xyxy[0]),int(xyxy[1])),(int(xyxy[2]),int(xyxy[3]))
                # 左上角坐标(c1[0],c1[1])
                # 右上角坐标(c2[0],c1[1])
                # 左下角坐标(c1[0],c2[1])
                # 右下角坐标(c2[0],c2[1])
                # 经过一个xyxy2xywh函数将左上角,右上角坐标转化为中心坐标以及宽和高
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()
                xywh = [round(x) for x in xywh]
                xywh = [
                    xywh[0] - xywh[2] // 2, xywh[1] - xywh[3] // 2, xywh[2],
                    xywh[3]
                ]  # 检测到目标位置（x,y,w,h）
                cls = names[int(cls)] # 类别
                conf = float(conf) # 置信度
                detections.append({
                    'class': cls,
                    'conf': conf,
                    'position': xywh
                })
    if len(detections) == 0:
        raise Exception('locate error')
    # 在高度和宽度上分别增加10将车牌图像可以完全截取
    img_cut = img[xywh[1]-5:xywh[1] + xywh[3] + 5, xywh[0]-10:xywh[0] + xywh[2] + 10]

    # 画出预测框
    img_copy = im0.copy()
    label =  f'{names[0]} {conf:.2f}'
    line_thickness = 2 # 宽度
    annotator = Annotator(img_copy, line_width=line_thickness, example=str(names))
    annotator.box_label(xyxy, label, color=colors(0, True))
    img_copy = annotator.result()
    return img_cut, img_copy
