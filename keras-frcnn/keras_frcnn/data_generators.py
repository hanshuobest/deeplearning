#coding:utf-8
from __future__ import absolute_import
import numpy as np
import cv2
import random
import copy
from . import data_augment
import threading
import itertools


def union(au, bu, area_intersection):
	'''
	计算并
	:param au:
	:param bu:
	:param area_intersection:
	:return:
	'''
	area_a = (au[2] - au[0]) * (au[3] - au[1])
	area_b = (bu[2] - bu[0]) * (bu[3] - bu[1])
	area_union = area_a + area_b - area_intersection
	return area_union


def intersection(ai, bi):
	'''
	计算交
	:param ai:
	:param bi:
	:return:
	'''
	x = max(ai[0], bi[0])
	y = max(ai[1], bi[1])
	w = min(ai[2], bi[2]) - x
	h = min(ai[3], bi[3]) - y
	if w < 0 or h < 0:
		return 0
	return w*h


def iou(a, b):
	'''
	计算iou系数
	:param a: (x1,y1,x2,y2)
	:param b: (x1,y1,x2,y2)
	:return:返回iou系数
	'''
	# a and b should be (x1,y1,x2,y2)

	if a[0] >= a[2] or a[1] >= a[3] or b[0] >= b[2] or b[1] >= b[3]:
		return 0.0

	area_i = intersection(a, b)
	area_u = union(a, b, area_i)

	return float(area_i) / float(area_u + 1e-6)


def get_new_img_size(width, height, img_min_side=600):
	'''
	计算新图片的尺寸
	:param width:
	:param height:
	:param img_min_side:新图片最小边的尺寸
	:return:
	'''
	if width <= height:
		f = float(img_min_side) / width
		resized_height = int(f * height)
		resized_width = img_min_side
	else:
		f = float(img_min_side) / height
		resized_width = int(f * width)
		resized_height = img_min_side

	return resized_width, resized_height


class SampleSelector:
	'''
	样本选择类
	'''
	def __init__(self, class_count):
		'''

		:param class_count: 类别字典
		:return:
		'''
		# ignore classes that have zero samples
		# 忽略有0个样本的类别
		self.classes = [b for b in class_count.keys() if class_count[b] > 0]
		self.class_cycle = itertools.cycle(self.classes)
		# 获取当前类别
		self.curr_class = next(self.class_cycle)

	def skip_sample_for_balanced_class(self, img_data):

		class_in_img = False

		for bbox in img_data['bboxes']:

			cls_name = bbox['class']

			if cls_name == self.curr_class:
				class_in_img = True
				self.curr_class = next(self.class_cycle)
				break

		if class_in_img:
			return False
		else:
			return True


def calc_rpn(C, img_data, width, height, resized_width, resized_height, img_length_calc_function):
	'''
	计算rpn
	:param C: 配置信息
	:param img_data: 包含一张图片的路径，bbox的坐标和对应的分类(一张图片可能有多组对象)
	:param width:
	:param height:
	:param resized_width:
	:param resized_height:
	:param img_length_calc_function:
	:return:y_rpn_cls,y_rpn_regr是否包含物体类别信息，和回归梯度,其形状为[1 , 2 * num_anchors , height , width] , [1 , 8 * num_anchors , height , width]
	'''

	# 图片到特征图的缩放倍数
	downscale = float(C.rpn_stride)
	# anchor 尺寸
	anchor_sizes = C.anchor_box_scales # 3
	# anchor 比率
	anchor_ratios = C.anchor_box_ratios# 3
	# anchor 的数量
	num_anchors = len(anchor_sizes) * len(anchor_ratios) # 9

	# calculate the output map size based on the network architecture
	# 基于网络结构计算输出特征图的尺寸
	(output_width, output_height) = img_length_calc_function(resized_width, resized_height)

	n_anchratios = len(anchor_ratios)
	
	# initialise empty output objectives
	# 初始化空的输出目标
	# y_rpn_overlap [output_height , output_width , 9] ， 三维张量
	y_rpn_overlap = np.zeros((output_height, output_width, num_anchors)) # output_height * output_width * num_anchors
	# y_is_box_valid [output_height , output_width , 9] 这个变量表示的是什么 ？三维张量
	y_is_box_valid = np.zeros((output_height, output_width, num_anchors))# output_height * output_width * num_anchors
	# y_rpn_regr [output_height , output_width , 36] 三维张量
	y_rpn_regr = np.zeros((output_height, output_width, num_anchors * 4))# output_height * output_width * 4 * num_anchors

	# 计算每张图片有多少个bbox
	num_bboxes = len(img_data['bboxes'])
	print('num_bboxes:' , num_bboxes)

	# 保存每个bbox对应的anchor有多少个属性为positive ， 是一个一维数组
	num_anchors_for_bbox = np.zeros(num_bboxes).astype(int)

	# 4 * 4 每一行表示为[jy ,jx , radio_idx , size_idx]，二维数组，所有元素默认为-1
	best_anchor_for_bbox = -1*np.ones((num_bboxes, 4)).astype(int) # num_bboxes * 4

	# 记录每个bbox对应的iou系数，一维数组
	best_iou_for_bbox = np.zeros(num_bboxes).astype(np.float32)

	# 保存anchor box坐标 [x1 , x2 , y1 , y2]
	best_x_for_bbox = np.zeros((num_bboxes, 4)).astype(int)

	# 保存anchor 偏移 [tx , ty , tw, th]
	best_dx_for_bbox = np.zeros((num_bboxes, 4)).astype(np.float32)

	# get the GT box coordinates, and resize to account for image resizing
	# 特征图上GT的坐标[x1 , x2 , y1 , y2]
	gta = np.zeros((num_bboxes, 4))
	for bbox_num, bbox in enumerate(img_data['bboxes']):
		# get the GT box coordinates, and resize to account for image resizing
		gta[bbox_num, 0] = bbox['x1'] * (resized_width / float(width))
		gta[bbox_num, 1] = bbox['x2'] * (resized_width / float(width))
		gta[bbox_num, 2] = bbox['y1'] * (resized_height / float(height))
		gta[bbox_num, 3] = bbox['y2'] * (resized_height / float(height))
	
	# rpn ground truth
	for anchor_size_idx in range(len(anchor_sizes)):
		for anchor_ratio_idx in range(n_anchratios):
			anchor_x = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][0] # anchor 的宽度
			anchor_y = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][1] # anchor 的高度
			
			for ix in range(output_width):					
				# x-coordinates of the current anchor box
				# 当前anchor box的x坐标 （x + 0.5）* 尺度 - anchor_x / 2
				x1_anc = downscale * (ix + 0.5) - anchor_x / 2
				x2_anc = downscale * (ix + 0.5) + anchor_x / 2	
				
				# ignore boxes that go across image boundaries
				# 忽略掉超过图像边界的点
				if x1_anc < 0 or x2_anc > resized_width:
					continue
					
				for jy in range(output_height):

					# y-coordinates of the current anchor box
					y1_anc = downscale * (jy + 0.5) - anchor_y / 2
					y2_anc = downscale * (jy + 0.5) + anchor_y / 2

					# ignore boxes that go across image boundaries
					if y1_anc < 0 or y2_anc > resized_height:
						continue

					# bbox_type indicates whether an anchor should be a target 
					bbox_type = 'neg'

					# this is the best IOU for the (x,y) coord and the current anchor
					# note that this is different from the best IOU for a GT bbox
					best_iou_for_loc = 0.0

					for bbox_num in range(num_bboxes):
						
						# get IOU of the current GT box and the current anchor box
						# 计算iou系数 ，输入的参数坐标形式[x1,y1,x2,y2]
						curr_iou = iou([gta[bbox_num, 0], gta[bbox_num, 2], gta[bbox_num, 1], gta[bbox_num, 3]], [x1_anc, y1_anc, x2_anc, y2_anc])
						# calculate the regression targets if they will be needed
						if curr_iou > best_iou_for_bbox[bbox_num] or curr_iou > C.rpn_max_overlap:
							# bbox中心点坐标
							cx = (gta[bbox_num, 0] + gta[bbox_num, 1]) / 2.0
							cy = (gta[bbox_num, 2] + gta[bbox_num, 3]) / 2.0
							# 真实bbox中心点坐标
							cxa = (x1_anc + x2_anc)/2.0
							cya = (y1_anc + y2_anc)/2.0

							# ground truth和proposal计算得到真正需要的平移量
							tx = (cx - cxa) / (x2_anc - x1_anc)
							ty = (cy - cya) / (y2_anc - y1_anc)
							tw = np.log((gta[bbox_num, 1] - gta[bbox_num, 0]) / (x2_anc - x1_anc))
							th = np.log((gta[bbox_num, 3] - gta[bbox_num, 2]) / (y2_anc - y1_anc))

						# 如果该bbox不是背景
						if img_data['bboxes'][bbox_num]['class'] != 'bg':
							# all GT boxes should be mapped to an anchor box, so we keep track of which anchor box was best
							# 查找哪一个anchor box最好
							if curr_iou > best_iou_for_bbox[bbox_num]:
								best_anchor_for_bbox[bbox_num] = [jy, ix, anchor_ratio_idx, anchor_size_idx]
								best_iou_for_bbox[bbox_num] = curr_iou
								best_x_for_bbox[bbox_num,:] = [x1_anc, x2_anc, y1_anc, y2_anc]
								best_dx_for_bbox[bbox_num,:] = [tx, ty, tw, th]

							# we set the anchor to positive if the IOU is >0.7 (it does not matter if there was another better box, it just indicates overlap)
							if curr_iou > C.rpn_max_overlap:
								bbox_type = 'pos'
								num_anchors_for_bbox[bbox_num] += 1
								# we update the regression layer target if this IOU is the best for the current (x,y) and anchor position
								if curr_iou > best_iou_for_loc:
									best_iou_for_loc = curr_iou
									best_regr = (tx, ty, tw, th)

							# if the IOU is >0.3 and <0.7, it is ambiguous and no included in the objective
							if C.rpn_min_overlap < curr_iou < C.rpn_max_overlap:
								# gray zone between neg and pos
								if bbox_type != 'pos':
									bbox_type = 'neutral' # 中性

					# turn on or off outputs depending on IOUs
					if bbox_type == 'neg':
						y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1
						y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
					elif bbox_type == 'neutral':
						y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
						y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
					elif bbox_type == 'pos':
						y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1
						y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1
						start = 4 * (anchor_ratio_idx + n_anchratios * anchor_size_idx)
						y_rpn_regr[jy, ix, start:start+4] = best_regr #元组

	# we ensure that every bbox has at least one positive RPN region
	# 确保每一个bbox至少有一个正RPN区域
	for idx in range(num_anchors_for_bbox.shape[0]):
		if num_anchors_for_bbox[idx] == 0: # 该bbox对应的anchor数量为0
			# no box with an IOU greater than zero ...
			if best_anchor_for_bbox[idx, 0] == -1:
				continue
			y_is_box_valid[
				best_anchor_for_bbox[idx,0], best_anchor_for_bbox[idx,1], best_anchor_for_bbox[idx,2] + n_anchratios *
				best_anchor_for_bbox[idx,3]] = 1
			y_rpn_overlap[
				best_anchor_for_bbox[idx,0], best_anchor_for_bbox[idx,1], best_anchor_for_bbox[idx,2] + n_anchratios *
				best_anchor_for_bbox[idx,3]] = 1
			start = 4 * (best_anchor_for_bbox[idx,2] + n_anchratios * best_anchor_for_bbox[idx,3])

			# best_anchor_for_bbox[idx,0] = y坐标 ，best_anchor_for_bbox[idx,1] = x坐标
			y_rpn_regr[
				best_anchor_for_bbox[idx,0], best_anchor_for_bbox[idx,1], start:start+4] = best_dx_for_bbox[idx, :]

	# 转为 [9 , output_height , output_width]
	y_rpn_overlap = np.transpose(y_rpn_overlap, (2, 0, 1))
	y_rpn_overlap = np.expand_dims(y_rpn_overlap, axis=0)


	# 转为 [9 , output_height , output_width]
	y_is_box_valid = np.transpose(y_is_box_valid, (2, 0, 1))
	y_is_box_valid = np.expand_dims(y_is_box_valid, axis=0)

	# 转为 [36 , output_height , output_width]
	y_rpn_regr = np.transpose(y_rpn_regr, (2, 0, 1))
	# 转为 [1 , 9 , output_height , output_width]
	y_rpn_regr = np.expand_dims(y_rpn_regr, axis=0)

	# np.where 返回符合条件的行号
	# pos_locs 正样本位置索引
	pos_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 1, y_is_box_valid[0, :, :, :] == 1))
	neg_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 0, y_is_box_valid[0, :, :, :] == 1))

	num_pos = len(pos_locs[0])

	# one issue is that the RPN has many more negative than positive regions, so we turn off some of the negative
	# regions. We also limit it to 256 regions.
	num_regions = 256

	if len(pos_locs[0]) > num_regions/2:
		# 从指定的序列中，随机的截取指定长度的片段，不作原地的修改
		val_locs = random.sample(range(len(pos_locs[0])), len(pos_locs[0]) - num_regions/2)
		y_is_box_valid[0, pos_locs[0][val_locs], pos_locs[1][val_locs], pos_locs[2][val_locs]] = 0
		num_pos = num_regions/2

	if len(neg_locs[0]) + num_pos > num_regions:
		val_locs = random.sample(range(len(neg_locs[0])), len(neg_locs[0]) - num_pos)
		y_is_box_valid[0, neg_locs[0][val_locs], neg_locs[1][val_locs], neg_locs[2][val_locs]] = 0

    # y_is_box_valid [1 , num_anchors , output_height , output_width]
    # y_rpn_overlap [1 , num_anchors , output_height , output_width]
	# y_rpn_cls [1 , 2 * num_anchors , output_height , output_width]
	y_rpn_cls = np.concatenate([y_is_box_valid, y_rpn_overlap], axis=1)
    # y_rpn_regr [1 , 4 * anchors , output_height , output_width]
	y_rpn_regr = np.concatenate([np.repeat(y_rpn_overlap, 4, axis=1), y_rpn_regr], axis=1)
    # y_rpn_regr [1 , 8 * anchors , output_height , output_width]

	print('y_rpn_cls:' , y_rpn_cls.shape)
	print('y_rpn_regr:' , y_rpn_regr.shape)
	return np.copy(y_rpn_cls), np.copy(y_rpn_regr)


class threadsafe_iter:
	"""Takes an iterator/generator and makes it thread-safe by
	serializing call to the `next` method of given iterator/generator.
	"""
	def __init__(self, it):
		self.it = it
		self.lock = threading.Lock()

	def __iter__(self):
		return self

	def next(self):
		with self.lock:
			return next(self.it)		

	
def threadsafe_generator(f):
	"""A decorator that takes a generator function and makes it thread-safe.
	"""
	def g(*a, **kw):
		return threadsafe_iter(f(*a, **kw))
	return g

def get_anchor_gt(all_img_data, class_count, C, img_length_calc_function, backend, mode='train'):
	'''
    生成器
	:param all_img_data: 训练图片列表
	:param class_count: 类别字典
	:param C:配置参数信息
	:param img_length_calc_function:计算图片尺寸的函数
	:param backend:
	:param mode:
	:return:
	'''

	# The following line is not useful with Python 3.5, it is kept for the legacy
	# all_img_data = sorted(all_img_data)

	# 样本选择器对象
	sample_selector = SampleSelector(class_count)
	while True:
		if mode == 'train':
			np.random.shuffle(all_img_data)

		for img_data in all_img_data:
			try:
				if C.balanced_classes and sample_selector.skip_sample_for_balanced_class(img_data):
					continue

				# read in image, and optionally add augmentation
				if mode == 'train':
                    # img_data_aug 增强后的图片
                    # x_img 原始图片集
					img_data_aug, x_img = data_augment.augment(img_data, C, augment=True)
				else:
					img_data_aug, x_img = data_augment.augment(img_data, C, augment=False)

				(width, height) = (img_data_aug['width'], img_data_aug['height'])
				(rows, cols, _) = x_img.shape

				assert cols == width
				assert rows == height

				# get image dimensions for resizing
				# 获取新图片尺寸
				(resized_width, resized_height) = get_new_img_size(width, height, C.im_size)

				# resize the image so that smalles side is length = 600px
				x_img = cv2.resize(x_img, (resized_width, resized_height), interpolation=cv2.INTER_CUBIC)

				try:
                    # y_rpn_cls [1 , 2 * num_anchors , height , width]
                    # y_rpn_regr [1 , 8 * num_anchors , height , width]
					y_rpn_cls, y_rpn_regr = calc_rpn(C, img_data_aug, width, height, resized_width, resized_height, img_length_calc_function)
				except:
					continue

				# Zero-center by mean pixel, and preprocess image
				# 图像预处理
				x_img = x_img[:,:, (2, 1, 0)]  # BGR -> RGB
				x_img = x_img.astype(np.float32)
				x_img[:, :, 0] -= C.img_channel_mean[0]
				x_img[:, :, 1] -= C.img_channel_mean[1]
				x_img[:, :, 2] -= C.img_channel_mean[2]
				x_img /= C.img_scaling_factor

				print('x_img.shape:' , x_img.shape)
				# x_img的尺寸变为(3 , 600 , 800)
				x_img = np.transpose(x_img, (2, 0, 1))
                # x_img [1 , 3 , height , width]
				x_img = np.expand_dims(x_img, axis=0)

				# y_rpn_regr.shape=(1 , 72 , 37 , 50)
				print('y_rpn_regr.shape[1]:' , y_rpn_regr.shape[1])
				y_rpn_regr[:, y_rpn_regr.shape[1]//2:, :, :] *= C.std_scaling # 后半部分为何要乘以标准尺度

				if backend == 'tf':
                    # x_img [1 , height , width , channels]
					x_img = np.transpose(x_img, (0, 2, 3, 1))
					# y_rpn_cls.shape=(1 , 18 , 37 , 50)
                    # y_rpn_cls.shape=(1 , height , width , 2 * num_anchors)
					y_rpn_cls = np.transpose(y_rpn_cls, (0, 2, 3, 1))
                    # y_rpn_cls.shape=(1 , height , width , 8 * num_anchors)
					y_rpn_regr = np.transpose(y_rpn_regr, (0, 2, 3, 1))

				yield np.copy(x_img), [np.copy(y_rpn_cls), np.copy(y_rpn_regr)], img_data_aug

			except Exception as e:
				print(e)
				continue
