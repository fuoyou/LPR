import re

import torch.nn as nn
import torch
import cv2
import numpy as np
from lprnet.model import *

CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
         '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
         '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
         '新',
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
         'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z', 'I', 'O', '-'
         ]
provinces = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
             '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
             '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
             '新']

# 图像预处理
def transform(img):
    img = img.astype('float32')
    img -= 127.5
    img *= 0.0078125
    img = np.transpose(img, (2, 0, 1))
    return img

def LPR_predict(img,lprnet):
    # 修改图像大小
    img = cv2.resize(img, (94, 24))
    im = transform(img)
    im = im[np.newaxis, :]
    ims = torch.Tensor(im)
    device = torch.device('cpu')
    # 预测网络
    prebs = lprnet(ims.to(device))
    prebs = prebs.cpu().detach().numpy()
    preb_labels = list()
    for i in range(prebs.shape[0]):
        preb = prebs[i, :, :]
        preb_label = list()
        for j in range(preb.shape[1]):
            preb_label.append(np.argmax(preb[:, j], axis=0))
        no_repeat_blank_label = list()
        pre_c = preb_label[0]
        if pre_c != len(CHARS) - 1:
            no_repeat_blank_label.append(pre_c)
        for c in preb_label:
            if (pre_c == c) or (c == len(CHARS) - 1):
                if c == len(CHARS) - 1:
                    pre_c = c
                continue
            no_repeat_blank_label.append(c)
            pre_c = c
        preb_labels.append(no_repeat_blank_label)

    # 输出检测结果
    plat = np.array(preb_labels)
    a = list()
    for i in range(0, plat.shape[1]):
        b = CHARS[plat[0][i]]
        a.append(b)
    if a != None:
        if a[0] not in provinces:
            a = a[1:]
            # 使用join方法将列表转换为字符串
            license = "".join(a)
        else:
            license = "".join(a)
        return license # 返回车牌号
    else:
        return "未能识别"

def LPR_predict_video(img,lprnet):
    # 修改图像大小
    img = cv2.resize(img, (94, 24))
    im = transform(img)
    im = im[np.newaxis, :]
    ims = torch.Tensor(im)
    device = torch.device('cpu')
    # 预测网络
    prebs = lprnet(ims.to(device))
    prebs = prebs.cpu().detach().numpy()
    preb_labels = list()
    for i in range(prebs.shape[0]):
        preb = prebs[i, :, :]
        preb_label = list()
        for j in range(preb.shape[1]):
            preb_label.append(np.argmax(preb[:, j], axis=0))
        no_repeat_blank_label = list()
        pre_c = preb_label[0]
        if pre_c != len(CHARS) - 1:
            no_repeat_blank_label.append(pre_c)
        for c in preb_label:
            if (pre_c == c) or (c == len(CHARS) - 1):
                if c == len(CHARS) - 1:
                    pre_c = c
                continue
            no_repeat_blank_label.append(c)
            pre_c = c
        preb_labels.append(no_repeat_blank_label)

    # 输出检测结果
    plat = np.array(preb_labels)
    a = list()
    for i in range(0, plat.shape[1]):
        b = CHARS[plat[0][i]]
        a.append(b)
    if a != None:
        license = "".join(a)
        return license # 返回车牌号
    else:
        return "未能识别"