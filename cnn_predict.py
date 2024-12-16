import cv2
import numpy as np
import tensorflow.keras as keras

# cnn = keras.models.load_model('../cnn/weights/cnn.h5')


def cnn_blue_predict(img,cnn_blue):
    characters = ["京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "皖", "闽", "赣", "鲁", "豫",
                  "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "0", "1", "2",
                  "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M",
                  "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
    img = cv2.resize(img, (240, 80))
    lic_prediction = cnn_blue.predict(img.reshape(1, 80, 240, 3))  # 预测形状应为(1,80,240,3)
    lic_prediction = np.array(lic_prediction).reshape(7, 65)  # 列表转为ndarray，形状为(7,65)
    confidence = calculate_confidence(lic_prediction)
    if len(lic_prediction[lic_prediction >= 0.8]) >= 4:  # 统计其中预测概率值大于80%以上的个数，大于等于4个以上认为识别率高，识别成功
        chars = ''
        for arg in np.argmax(lic_prediction, axis=1):  # 取每行中概率值最大的arg,将其转为字符
            chars += characters[arg]
        return chars,confidence
    else:
        return None,None


def cnn_green_predict(img,cnn_green):
    characters = ["京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "皖", "闽", "赣", "鲁", "豫",
                  "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "0", "1", "2",
                  "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M",
                  "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

    img = cv2.resize(img,(240, 80))
    lic_prediction = cnn_green.predict(img.reshape(1, 80, 240, 3))  # 预测形状应为(1,80,240,3)
    lic_prediction = np.array(lic_prediction).reshape(8, 65)  # 列表转为ndarray，形状为(7,65)
    confidence = calculate_confidence(lic_prediction)
    if len(lic_prediction[lic_prediction >= 0.8]) >= 4:  # 统计其中预测概率值大于80%以上的个数，大于等于4个以上认为识别率高，识别成功
        chars = ''
        for arg in np.argmax(lic_prediction, axis=1):  # 取每行中概率值最大的arg,将其转为字符
            chars += characters[arg]
        return chars,confidence # 返回车牌号码以及置信度
    else:
        return None,None

# 统计CNN模型的总置信度
def calculate_confidence(lic_prediction):
    confidence_sum = 0
    num_chars = 0


    for i in range(lic_prediction.shape[0]):
        # 计算第i个字符的置信度
        confidence_i = np.max(lic_prediction[i])
        confidence_sum += confidence_i
        num_chars += 1

    # 计算平均置信度
    if num_chars > 0:
        avg_confidence = confidence_sum / num_chars
    else:
        avg_confidence = 0
    formatted_confidence = "{:.5f}".format(avg_confidence)
    return formatted_confidence
