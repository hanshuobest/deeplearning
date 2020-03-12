import os , sys

psenet_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)) , "..")
if os.path.exists(psenet_dir):
    print('exists')
sys.path.append(psenet_dir)
from psenet import PSENet , PSENetHandel
from utils import crop_rect
from crnn import CRNNHandle , FullCrnn , LiteCrnn
from angle_class import AangleClassHandle , shufflenet_v2_x0_5

#psenet相关
pse_long_size = 1280 #图片长边
pse_model_type  = "mobilenetv2"
pse_scale = 1

if pse_model_type == "mobilenetv2" :
    pse_model_path = "../models/psenet_lite_mbv2.pth"

if pse_model_type == "mobilenetv2" :
    text_detect_net = PSENet(backbone = pse_model_type , pretrained=False , result_num = 6 , scale = pse_scale)

text_handle = PSENetHandel(pse_model_path , text_detect_net , pse_scale)

# crnn相关
nh = 256
crnn_type  = "lite_lstm"
crnn_vertical_model_path = "../models/crnn_dw_lstm_vertical.pth"

if crnn_type == "lite_lstm":
    LSTMFLAG = True
    crnn_model_path =  ("../models/crnn_lite_lstm_dw.pth")
elif crnn_type == "lite_dense":
    LSTMFLAG = False
    crnn_model_path = "../models/crnn_lite_dense_dw.pth"
elif crnn_type == "full_lstm":
    LSTMFLAG = True
    crnn_model_path = "../models/ocr-lstm.pth"
elif crnn_type == "full_dense":
    LSTMFLAG = True
    crnn_model_path = "../models/ocr-dense.pth"

from crnn.keys import  alphabetChinese as alphabet

#angle_class相关
lable_map_dict  =  { 0 : "hengdao",  1:"hengzhen",  2:"shudao",  3:"shuzhen"} #hengdao: 文本行横向倒立 其他类似
rotae_map_dict  =   {"hengdao": 180 , "hengzhen": 0 , "shudao": 180 , "shuzhen": 0 } # 文本行需要旋转的角度
angle_type  = "shufflenetv2_05"
angle_model_path  =  "../models/{}.pth".format(angle_type)

crnn_net = None
if crnn_type == "full_lstm" or crnn_type == "full_dense":
    crnn_net  = FullCrnn(32, 1, len(alphabet) + 1, nh, n_rnn=2, leakyRelu=False, lstmFlag=LSTMFLAG)
elif crnn_type == "lite_lstm" or crnn_type == "lite_dense":
    crnn_net =  LiteCrnn(32, 1, len(alphabet) + 1, nh, n_rnn=2, leakyRelu=False, lstmFlag=LSTMFLAG)

assert  crnn_type is not None
crnn_handle  =  CRNNHandle(crnn_model_path , crnn_net)

crnn_vertical_handle = None
if crnn_vertical_model_path is not None:
    crnn_vertical_net = LiteCrnn(32, 1, len(alphabet) + 1, nh, n_rnn=2, leakyRelu=False, lstmFlag=True)
    crnn_vertical_handle = CRNNHandle(crnn_vertical_model_path , crnn_vertical_net)

assert angle_type in ["shufflenetv2_05"]
if angle_type == "shufflenetv2_05":
    angle_net = shufflenet_v2_x0_5(num_classes=len(lable_map_dict), pretrained=False)

angle_handle = AangleClassHandle(angle_model_path,angle_net)