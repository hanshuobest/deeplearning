import numpy as np
import glob
import os
import xml.etree.ElementTree as ET

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
    for i in range(1 ,len(label_count)):
        name = label_count[i].find('name').text
        print('name:' , name)
        points = label_count[i].find('points').findall('point')
        defect_info = {}
        defect_info['class'] = name
        polygon = []
        for point in points:
            x = float(point.find('x').text)
            y = float(point.find('y').text)
            polygon.append((x , y))
        defect_info['points'] = polygon
        defects_info.append(defect_info)
    fengji_info['defect'] = defects_info
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

all_defect_info = fengji_info['defect']
print('该图片一共有{}个缺陷'.format(len(all_defect_info)))
for defect_info in all_defect_info:
    print('缺陷类型：' , defect_info['class'])
    print('缺陷坐标信息：')
    for pt in defect_info['points']:
        print(pt)

