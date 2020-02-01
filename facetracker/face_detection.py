# 利用sort跟踪人脸

import dlib
import cv2
import time
import numpy as np
from sort import *

if __name__ == '__main__':
    # 使用dlib自带的前脸检测器作为我们的特征提取器
    detector = dlib.get_frontal_face_detector()
    cap = cv2.VideoCapture(0)
    tracker = Sort()
    need_detect_flag = True

    while cap.isOpened():
        ret , frame = cap.read()
        if ret:
            start_tt = time.time()
            frame = cv2.flip(frame , 1);
            frame = cv2.resize(frame , (640 , 480))

            dets = detector(frame , 1)
            print('dets: ' , dets)
            tracker_dets = []
            for i , d in enumerate(dets):
                cv2.rectangle(frame , tuple([d.left() , d.top()]) , tuple([d.right() , d.bottom()]) , (0 , 0 , 255) , 2)
                tracker_dets.append([d.left() , d.top() , d.right() , d.bottom()])

            # print(tracker_dets)
            track_bbs_ids = tracker.update(np.asarray(tracker_dets))
            print('track_bbs_ids: ' , track_bbs_ids)
            if len(track_bbs_ids) > 0:
                for dd in track_bbs_ids:
                    cv2.rectangle(frame , (int(dd[0]) , int(dd[1])) , (int(dd[2]) , int(dd[3])) , (0 , 255 , 0))
                    cv2.putText(frame , str(dd[4]) , (int((dd[0] + dd[2])//2) , int(dd[1])) , cv2.FONT_HERSHEY_SIMPLEX , 0.6 , (0 , 0 , 255))

            cost_time = (time.time() - start_tt) * 1000
            # print('cost time:' , cost_time)
            cv2.putText(frame , str(1000//cost_time) , (10 , 30) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (255 , 0 , 0))

            cv2.imshow("frame" , frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
