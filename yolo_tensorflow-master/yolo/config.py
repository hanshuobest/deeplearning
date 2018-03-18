import os

#
# path and dataset parameter
#

# 保存训练数据文件夹
DATA_PATH = 'data'

# data/pascal_voc
PASCAL_PATH = os.path.join(DATA_PATH, 'pascal_voc')
# data/cache
CACHE_PATH = os.path.join(PASCAL_PATH, 'cache')
# data/output
OUTPUT_DIR = os.path.join(PASCAL_PATH, 'output')
# data/weight
WEIGHTS_DIR = os.path.join(PASCAL_PATH, 'weight')

WEIGHTS_FILE = None
# WEIGHTS_FILE = os.path.join(DATA_PATH, 'weights', 'YOLO_small.ckpt')

CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
           'train', 'tvmonitor']

FLIPPED = True


#
# model parameter
#

IMAGE_SIZE = 448

# 将图像分成7*7的格子
CELL_SIZE = 7

# 每个格子使用两个box
BOXES_PER_CELL = 2

ALPHA = 0.1

DISP_CONSOLE = False

OBJECT_SCALE = 1.0
NOOBJECT_SCALE = 1.0
CLASS_SCALE = 2.0
COORD_SCALE = 5.0


#
# solver parameter
#

GPU = ''

LEARNING_RATE = 0.0001

DECAY_STEPS = 30000

DECAY_RATE = 0.1

STAIRCASE = True

BATCH_SIZE = 2

MAX_ITER = 15000

SUMMARY_ITER = 10

SAVE_ITER = 1000


#
# test parameter
#

THRESHOLD = 0.2

IOU_THRESHOLD = 0.5
