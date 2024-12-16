import tensorflow.keras as keras
import cv2


# unet = keras.models.load_model('../Unet/weights/locate.h5')
# 返回的是一个二值图像掩码,其中车牌区域为白色其他背景为黑色
def unet_predict(unet,img):
    H,W = img.shape[:2]
    img = cv2.resize(img, (512, 512))
    img = img.reshape(1, 512, 512, 3)
    img_mask = unet.predict(img)
    # (1,512,512,3)
    # print(img_mask.shape)
    # numpy.array类型
    # print(type(img_mask))
    image = img.reshape(H,W,3)
    return image,img_mask
