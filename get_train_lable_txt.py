# -*- coding: utf-8 -*-
"""
使用六个特征点计算出欧拉角，然后转换为pose分，再写入txt文件中
形成完整的数据
bbox x y w h 6_ld l_eye_x l_eye_y ..... score
"""


import os
import threading

import cv2
import math
import numpy as np
from math import cos,sin
from tools import getBestChin
import tqdm

model_points = np.array([
        (0.0, 0.0, 0.0),  # Nose tip
        (0.0, -330.0, -65.0),  # Chin
        (-165.0, 170.0, -135.0),  # Left eye left corner
        (165.0, 170.0, -135.0),  # Right eye right corne
        (-150.0, -150.0, -125.0),  # Left Mouth corner
        (150.0, -150.0, -125.0)  # Right mouth corner
    ])

def pose_estimate(txt_path):
    with open(f'../6_train_txt_center/{txt_path}', 'r') as f:
        with open(f'../6_lables_txt_center/{txt_path}', 'w') as wf:
            lines = f.readlines()
            # img_name = txt_path.replace(".ttt",".jpg")
            # img = cv2.imread(f'../coco_val/{img_name}')

            # h,w = img.shape[0:2]
            # focal_length = w / 2 / math.tan((60 / 2) * (math.pi / 180))
            # center = (w // 2, h // 2)
            # dist_coeffs = np.zeros((4, 1))
            # camera_matrix = np.array([
            #     (focal_length,0,center[0]),
            #     (0,focal_length,center[1]),
            #     (0,0,1)
            # ])
            for line in lines:
                data = line.split()
                points = data[6:]
                # 数据为：
                bbox = list(map(float,data[1:5]))
                bbox = list(map(int,bbox))
                # # Nose tip
                # (0.0, -330.0, -65.0),  # Chin
                # (-225.0, 170.0, -135.0),  # Left eye left corner
                # (225.0, 170.0, -135.0),  # Right eye right corne
                # (-150.0, -150.0, -125.0),  # Left Mouth corner
                # (150.0, -150.0, -125.0)  # Right mouth corner
                x = list(map(float, points[::2]))
                y = list(map(float, points[1::2]))

                left_eye = [x[0], y[0]]
                right_eye = [x[1],y[1]]
                nose = [x[2], y[2]]
                left_mouse = [x[3], y[3]]
                right_mouse = [x[4], y[4]]
                chin = [x[5],y[5]]

                # # 无效人脸,作为负样本
                # if left_eye[0] == -1 and chin[0] == -1:
                #     score = 0.0
                # else:
                #     image_points = np.array([
                #         nose,  # Nose tip
                #         chin,  # chin
                #         left_eye,  # Left eye left corner
                #         right_eye,  # Right eye right corne
                #
                #         left_mouse,  # Left Mouth corner
                #         right_mouse  # Right mouth corner
                #     ], dtype="double")
                #     (_, rvec, tvec) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs,
                #                                    flags=cv2.SOLVEPNP_UPNP)
                #     # Calculate Euler angles
                #     rmat = cv2.Rodrigues(rvec)[0]  # rotation matrix
                #     pmat = np.hstack((rmat, tvec))  # projection matrix
                #     eulers = cv2.decomposeProjectionMatrix(pmat)[-1]
                #
                #
                #     pitch, yaw, roll = [math.radians(_) for _ in eulers]
                #     pitch = -math.degrees(math.asin(math.sin(pitch)))
                #     roll = -math.degrees(math.asin(math.sin(roll)))
                #     yaw = math.degrees(math.asin(math.sin(yaw)))
                #     score = max(1 - abs(yaw) / 80 * 0.4 - abs(pitch) / 90 * 0.3 - abs(roll) / 90 * 0.3, 0.0)

                tmp_lst = data[1:5]
                tmp_lst += left_eye
                tmp_lst += right_eye
                tmp_lst += nose
                tmp_lst += left_mouse
                tmp_lst += right_mouse
                # tmp_lst.append(str(score))
                tmp_lst = [str(x) for x in tmp_lst]
                wf.write(" ".join(tmp_lst) + '\n')



def thread(txt_lst):
    for txt in tqdm.tqdm(txt_lst):
        pose_estimate(txt)
if __name__ == '__main__':
    # pose_estimate('000000002477.ttt')
    n_thread = 6
    txt_lst = os.listdir('../6_train_txt_center')
    len_evey = len(txt_lst) // n_thread
    Processes = []
    for i in range(6+1):
        if i < 6:
            p = threading.Thread(target=thread, args=(txt_lst[i*len_evey:(i+1)*len_evey],))
        else:
            p = threading.Thread(target=thread, args=(txt_lst[i * len_evey:],))
        Processes.append(p)


    for p in Processes:
        p.start()
    for p in Processes:
        p.join()
