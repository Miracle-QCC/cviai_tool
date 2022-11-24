#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import glob

import numpy as np
import tqdm

from get_elur_angle import *
from get_elur_angle import _radian2angle

def get_dis(p1,p2):
    return np.sqrt(((p1 - p2) ** 2).sum())

def run(root):
    imgs_lst = glob.glob(root + '/*jpg')
    imgs_lst.sort()
    txts_lst = glob.glob(root + '/*ttt')
    txts_lst.sort()
    n = len(imgs_lst)
    for i in tqdm.tqdm(range(n)):
        img_path = imgs_lst[i]
        txt_path = txts_lst[i]
        img = cv2.imread(img_path)
        with open(txt_path, 'r') as f:
            data = f.readline().split()
        img_name = img_path.split('\\')[-1]
        if not data:
            print(img_path.split('\\')[-1], "无人脸")
            cv2.imwrite(r"D:\benchmark\no_face\%s" % img_name, img)
            continue

        bbox = data[:4]
        bbox = list(map(int,bbox))
        w,h = bbox[2] - bbox[0],bbox[3] - bbox[1]

        area_score = min(1.0,w*h / (112*112))

        points_5 = data[5:]
        points_5 = list(map(float,points_5))
        points_5 = np.array([points_5])
        points_5 = points_5.reshape(5,2)

        l_e = points_5[0]
        r_e = points_5[1]
        nose = points_5[2]
        l_m = points_5[3]
        r_m = points_5[4]



        if nose[0] < min(l_e[0],l_m[0]) or nose[0] > max(r_e[0],r_m[0]):
            score = 0.0
        # 如果左眼在右眼的右， 左嘴在右嘴的右，则赋值0
        elif l_e[0] > r_e[0] or l_m[0] > r_m[0]:
            score = 0.0
        # 如果眼睛或嘴巴的两个点挤在一块
        elif get_dis(l_e,r_e) < 0.25 * w or get_dis(l_m,r_m) < 0.2 * w:
            score = 0.0

        else:
            pose_estimator = PoseEstimator(img_size=img.shape[:2])
            pose = pose_estimator.solve_pose_by_5_points(points_5)
            pitch, yaw, roll = pose_estimator.get_euler_angle(pose[0])



            Y, X, Z = map(_radian2angle, [pitch, yaw, roll])
            # line = 'Y:{:.1f}\nX:{:.1f}\nZ:{:.1f}'.format(Y, X, Z)
            score = 1 - min(1,0.4 * abs(X) / 90 + 0.3 * abs(Y) / 90 + 0.3*abs(Z) / 90)

            score *= area_score

        cv2.putText(img, str(score), (bbox[0],bbox[1]), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 255), 2)
        print(img_name, "得分为:%.4f"%score)

        cv2.imwrite(r"D:\benchmark\with_face\%s"%img_name, img)
        # for _, ttt in enumerate(line.split('\n')):
        #     cv2.putText(img, ttt, (20, y), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 255), 1)
        #     y = y + 15

        # for p in points_5:
        #     cv2.circle(img, (int(p[0]), int(p[1])), 2, (0, 255, 0), -1, 0)

        # for p in points_68:
        #     cv2.circle(img, (int(p[0]), int(p[1])), 2, (0, 255, 0), -1, 0)

        # cv2.imshow('img', img)
        # if cv2.waitKey(-1) == 27:
        #     pass
        #
        # return 0


if __name__ == "__main__":
    root = r'D:\Code\scrfd\outputs'
    run(root)