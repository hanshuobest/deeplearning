from PIL import Image
from model import  text_predict,crnn_handle
import numpy as np
import cv2


def text(cv_img):
    result = text_predict(cv_img)
    print(result)

if __name__ == "__main__":
    imageName = "test_imgs/1.jpg"
    text(cv2.imread(imageName))

    # res = crnn_handle.predict(Image.open(imageName))
    # print(res)