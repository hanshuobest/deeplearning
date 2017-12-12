import numpy as np
import glob
import os
import xml.etree.ElementTree as ET
import math

def read_annotation(path):
    '''
    读取标准信息
    :param path: xml文件路径
    :return: 返回单张图像的标注信息 dict
    '''
    fengji_info = {}
    et = ET.parse(path)
    element_root = et.getroot()
    filename = element_root.find('filename').text

    labels = element_root.find('labels')
    label_count = labels.findall('label')

    color = label_count[0].find('color').text
    fengji_info['filename'] = filename
    fengji_info['color'] = color

    poly_points = label_count[0].find('points').findall('point')

    # 保存叶片轮廓信息
    polygon_lst = []
    for point in poly_points:
        x = float(point.find('x').text)
        y = float(point.find('y').text)
        polygon_lst.append((x , y))
    fengji_info['polygon'] = polygon_lst

    # 保存所有缺陷信息
    defects_info = []
    boxes = np.zeros((len(label_count) - 1 , 4) , dtype=np.uint16)
    gt_class = np.zeros((len(label_count) - 1) , dtype=np.int32)

    for i in range(1 ,len(label_count)):
        name = label_count[i].find('name').text
        print('name:' , name)
        points = label_count[i].find('points').findall('point')
        defect_info = {}
        defect_info['class'] = name
        polygon = []

        xmin = 10000
        ymin = 10000
        xmax = -10000
        ymax = -10000

        for point in points:
            x = float(point.find('x').text)
            y = float(point.find('y').text)
            polygon.append((x , y))

            xmax = max(x , xmax)
            ymax = max(y , ymax)
            xmin = min(x , xmin)
            ymin = min(y , ymin)
        boxes[i - 1 , :] = [xmin , ymin , xmax , ymax]

        if name == '表面不良':
            gt_class[i - 1] = 1
        elif name == '砂眼':
            gt_class[i - 1] = 2
        elif name == '涂料损伤':
            gt_class[i - 1] = 3
        else:
            gt_class[i - 1] = 4

    fengji_info['boxes'] = boxes
    fengji_info['gt_classes'] = gt_class
    return fengji_info

annotations_dir = "F:\\python\\deeplearning.git\\trunk\\Faster-RCNN-TensorFlow-Python3.5\\data\\CompanyData\\Train\\Annotations"
anno_lists = glob.glob(annotations_dir + "\\*.xml")
print(anno_lists)

file_object = open(annotations_dir +  '\\annotations.txt' , 'w')
for anno in anno_lists:
    file_object.write(anno + '\n')
file_object.close()

print('finished!')

read_object = open(annotations_dir + '\\annotations.txt' , 'r')
xml_files = []
try:
    line_texts = read_object.readlines()
    for line in line_texts:
        xml_files.append(line.strip())
finally:
    read_object.close()

fengji_info = read_annotation(xml_files[0])

print('该叶片名称：' , fengji_info['filename'])
print('该叶片颜色：' , fengji_info['color'])
print('该叶片轮廓信息：')
polygon_info = fengji_info['polygon']
for xy in polygon_info:
    print(xy)



all_boxes = fengji_info['boxes']
m , n = all_boxes.shape
print('m:{} ,n:{}'.format(m , n))
print(all_boxes)

gt_classes = fengji_info['gt_classes']
print(gt_classes)

print(fengji_info)
