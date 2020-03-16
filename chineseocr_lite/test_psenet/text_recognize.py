import os
import cv2
import numpy as np
import sys
from PIL import  Image
from model_config import *
import imutils

def draw_bbox(img_path, result, color=(255, 0, 0),thickness=2):
    if isinstance(img_path, str):
        img_path = cv2.imread(img_path)
    img_path = img_path.copy()
    for point in result:
        point = point.astype(int)
        cv2.line(img_path, tuple(point[0]), tuple(point[1]), color, thickness)
        cv2.line(img_path, tuple(point[1]), tuple(point[2]), color, thickness)
        cv2.line(img_path, tuple(point[2]), tuple(point[3]), color, thickness)
        cv2.line(img_path, tuple(point[3]), tuple(point[0]), color, thickness)
    return img_path

def crnnRec(im, rects_re, leftAdjust=False, rightAdjust=False, alph=0.2, f=1.0):
    """
    crnn模型，ocr识别
    """
    results = []
    im = Image.fromarray(im)
    for index, rect in enumerate(rects_re):
        degree, w, h, cx, cy = rect
        partImg = crop_rect(im,  ((cx, cy ),(h, w),degree))
        newW,newH = partImg.size
        partImg_array  = np.uint8(partImg)

        if newH > 1.5* newW:
            partImg_array = np.rot90(partImg_array,1)

        angel_index = angle_handle.predict(partImg_array)

        angel_class = lable_map_dict[angel_index]
        rotate_angle = rotae_map_dict[angel_class]

        if rotate_angle != 0 :
            partImg_array = np.rot90(partImg_array,rotate_angle//90)
        
        partImg = Image.fromarray(partImg_array).convert("RGB")
        
        partImg_ = partImg.convert('L')

        try:

            if crnn_vertical_handle is not None and angel_class in ["shudao", "shuzhen"]:
                simPred =  crnn_vertical_handle.predict(partImg_)
            else:
                simPred = crnn_handle.predict(partImg_)  ##识别的文本
        except :
            continue

        if simPred.strip() != u'':
            results.append({'cx': cx * f, 'cy': cy * f, 'text': simPred, 'w': newW * f, 'h': newH * f,
                            'degree': degree })
    return results

def text_predict(img , draw_flag = False):
    preds, boxes_list, rects_re, t = text_handle.predict(img, long_size=pse_long_size)


    result = crnnRec(img , rects_re)
    if draw_flag:
        img2 = draw_bbox(img, boxes_list, color=(0, 255, 0))
        return result, img2
    
    return result , None



if __name__ == "__main__":
    imgName = "../test_imgs/1.jpg"
    res , img = text_predict(cv2.imread(imgName) , True)
    print(res)
    cv2.imshow("img", img)
    cv2.waitKey(0)
    


    



