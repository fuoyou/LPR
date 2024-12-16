import cv2
import numpy as np
import torch
from torchvision.transforms import transforms
import random
# 画出轮廓传入参数 x==>左上角和右上角的坐标
from tools.operate import get_license_plate_color
from tools.yolov7_predict import getModel, findContoursAndDrawBoundingBox, skew_detection


def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 3  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)



# 对经过u2net截取的图片进行校正
def locate_correct(img_mask,origin_img):
    # 寻找轮廓
    contours, _ = cv2.findContours(img_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 找到最大轮廓
    max_area = 0
    max_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour
    # 获取最大轮廓的边界矩形
    x, y, w, h = cv2.boundingRect(max_contour)  # (75, 309, 149, 102)
    img_cut_mask = img_mask[y-5:y + h+5, x+5:x + w-5]  # 将标签车牌区域截取出来
    # 车牌区域的均值应较高，且宽和高不会非常小，因此通过以下条件进行筛选
    if np.mean(img_cut_mask) >= 75 and w > 15 and h > 15:
        # rect==>((149.77975463867188, 358.2179870605469),(170.74542236328125, 53.577552795410156),19.68332862854004)
        rect = cv2.minAreaRect(max_contour)  # 针对坐标点获取带方向角的最小外接矩形,中心点坐标,宽高,旋转角度
        # box ==> [[60 354]
        #          [78 304]
        #          [239 361]
        #          [221 412]]
        box = cv2.boxPoints(rect).astype(np.int32)  # 获取最小外接矩形四个顶点坐标
        cont = max_contour.reshape(-1, 2).tolist()  # 将轮廓转换为一个列表cont,每个元素是一个包含两个整数的列表,表示轮廓上的一个点的坐标
        # 排序依据是列表中每个元素的第一个元素，即坐标的x值
        # 排序后 box ==> [[60,354]
        #                [78,304]]
        #                [221,412]]
        #                [239,361]]
        box = sorted(box, key=lambda xy: xy[0])  # 先按照左右进行排序，分为左侧的坐标和右侧的坐标
        # box_left ==> [[60,354],
        #               [78,304]]
        # box_right ==> [[221,412],
        #               [239,361]]
        box_left, box_right = box[:2], box[2:]  # 此时box的前2个是左侧的坐标,后2个是右侧的坐标
        # 排序后box_left==> [[78,304],
        #                   [60,354]]
        box_left = sorted(box_left, key=lambda x: x[1])  # 再按照上下即y进行排序,此时box_left中为左上和左下两个端点坐标
        # 排序后box_left==> [[239,361],
        #                   [221,412]]
        box_right = sorted(box_right, key=lambda x: x[1])  # 此时box_right中为右上和右下两个端点坐标
        box = np.array(box_left + box_right)  # [左上，左下，右上，右下]
        x0, y0 = box[0][0], box[0][1]  # 这里的4个坐标即为最小外接矩形的四个坐标，接下来需获取平行(或不规则)四边形的坐标
        x1, y1 = box[1][0], box[1][1]
        x2, y2 = box[2][0], box[2][1]
        x3, y3 = box[3][0], box[3][1]

        def point_to_line_distance(X, Y):
            if x2 - x0:
                k_up = (y2 - y0) / (x2 - x0)  # 斜率不为无穷大
                d_up = abs(k_up * X - Y + y2 - k_up * x2) / (k_up ** 2 + 1) ** 0.5
            else:  # 斜率无穷大
                d_up = abs(X - x2)
            if x1 - x3:
                k_down = (y1 - y3) / (x1 - x3)  # 斜率不为无穷大
                d_down = abs(k_down * X - Y + y1 - k_down * x1) / (k_down ** 2 + 1) ** 0.5
            else:  # 斜率无穷大
                d_down = abs(X - x1)
            return d_up, d_down

        d0, d1, d2, d3 = np.inf, np.inf, np.inf, np.inf
        l0, l1, l2, l3 = (x0, y0), (x1, y1), (x2, y2), (x3, y3)

        for each in cont:  # 计算cont中的坐标与矩形四个坐标的距离以及到上下两条直线的距离，对距离和进行权重的添加，成功计算选出四边形的4个顶点坐标
            x, y = each[0], each[1]
            dis0 = (x - x0) ** 2 + (y - y0) ** 2
            dis1 = (x - x1) ** 2 + (y - y1) ** 2
            dis2 = (x - x2) ** 2 + (y - y2) ** 2
            dis3 = (x - x3) ** 2 + (y - y3) ** 2
            d_up, d_down = point_to_line_distance(x, y)
            weight = 0.975
            if weight * d_up + (1 - weight) * dis0 < d0:  # 小于则更新
                d0 = weight * d_up + (1 - weight) * dis0
                l0 = (x, y)
            if weight * d_down + (1 - weight) * dis1 < d1:
                d1 = weight * d_down + (1 - weight) * dis1
                l1 = (x, y)
            if weight * d_up + (1 - weight) * dis2 < d2:
                d2 = weight * d_up + (1 - weight) * dis2
                l2 = (x, y)
            if weight * d_down + (1 - weight) * dis3 < d3:
                d3 = weight * d_down + (1 - weight) * dis3
                l3 = (x, y)

        p0 = np.float32([l0, l1, l2, l3])  # 左上角，左下角，右上角，右下角，p0和p1中的坐标顺序对应，以进行转换矩阵的形成
        p1 = np.float32([(0, 0), (0, 80), (240, 0), (240, 80)])  # 我们所需的长方形
        transform_mat = cv2.getPerspectiveTransform(p0, p1)  # 构成转换矩阵
        lic = cv2.warpPerspective(origin_img, transform_mat, (240, 80))  # 进行车牌矫正
        # 在原图像上画出轮廓
        cv2.drawContours(origin_img, [np.array([l0, l1, l3, l2])], -1, (0, 0, 255), 3)
        # 在原图像上画出轮廓
        #xyxy = (l0[0], l0[1], l3[0], l3[1])
        #plot_one_box(xyxy, origin_img, color=(0, 0, 255), label="license plate", line_thickness=3)
        return lic,origin_img # 返回提取后的车牌图像
    else: # 不存在车牌
        return None,origin_img
    # 初始化U2NET模型
    # # 定义了U2-Net模型，并加载之前保存的权重文件
    # model = u2net_full()
    # weights = torch.load(weights_path, map_location='cpu')
    # if "model" in weights:
    #     model.load_state_dict(weights["model"])
    # else:
    #     model.load_state_dict(weights)
    # # 将模型移动到设备上
    # model.to(device)
    # # 评估模式下,模型不会更新其权重参数,只是用来预测结果
    # model.eval()



def U2NET_predict(img,model):

    img_copy = img.copy()
    # 检查当前设备是否支持CUDA
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 将图像转换为张量,并且对图像进行缩放和标准化
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(320),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225))
    ])

    # 读取图片并转为RGB形式,cv2读取的格式是BGR格式
    origin_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = origin_img.shape[:2]
    img = data_transform(origin_img)
    # 将图像添加一个维度并将其放置在设备上
    img = torch.unsqueeze(img, 0).to(device)  # [C, H, W] -> [1, C, H, W]
    # 禁用梯度计算,以节省内存和加速推理
    with torch.no_grad():
        # 对图像进行预测
        pred = model(img)
        # 去除不必要的通道数
        pred = torch.squeeze(pred).to("cpu").numpy()  # [1, 1, H, W] -> [H, W]
        # 将数组的大小调整为输入图像的大小
        pred = cv2.resize(pred, dsize=(w, h), interpolation=cv2.INTER_LINEAR)
        # 将pred转化为numpy数组并将其转化为uint8
        pred = (pred * 255).astype(np.uint8)
        # 二值化
        img_mask = cv2.threshold(pred, 0, 255, cv2.THRESH_BINARY)[1]
        license_image,img_label = locate_correct(img_mask, img_copy)
        return license_image,img_label
