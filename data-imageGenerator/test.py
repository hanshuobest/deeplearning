# -*- coding= utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import glob
from keras.preprocessing.image import ImageDataGenerator , array_to_img , img_to_array, load_img

datagen = ImageDataGenerator(featurewise_center=False , samplewise_center=False , featurewise_std_normalization=False , samplewise_std_normalization=False , zca_whitening=False  , rotation_range=20 , zoom_range=0.1 , height_shift_range= 0.125 , width_shift_range=0.125 , horizontal_flip=True , vertical_flip=True)

# dir_path = 'F:\\c++\\BladeDefectRecognition\\trunk\\0-Src\\windTurbine\\SVM_Project\\positive'
# dir_path = 'F:\\c++\\BladeDefectRecognition\\trunk\\0-Src\\windTurbine\\SVM_Project\\negtive'
dir_path = 'F:\\c++\\BladeDefectRecognition\\trunk\\0-Src\\windTurbine\\SVM_Project\\test'
file_lists = glob.glob(dir_path + '/*.jpg')
print(file_lists)

img_num = len(file_lists)
print('img_num:' , img_num)

k = 0
for img_file in file_lists:
    img = load_img(img_file)
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)
    i = 0
    for batch in datagen.flow(x , batch_size=1 , save_to_dir=dir_path + '/save' , save_prefix=img_file , save_format='jpg'):
        i += 1
        if i == 10:
            break
# output_file = open(dir_path + '\\positive_files.txt' , "w")
# output_file = open(dir_path + '\\negtive_files.txt' , "w")
output_file = open(dir_path + '\\test_files.txt' , "w")
file_lists = glob.glob(dir_path + '/*.jpg')
for img_file in file_lists:
    output_file.write(img_file + '\n')
output_file.close()



