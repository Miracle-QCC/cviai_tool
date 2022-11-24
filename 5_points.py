
# -*- coding: utf-8 -*-
"""
根据coco_wholebody数据集的GTjson文件获得人脸的5个特征点，然后利用这些特征点来计算欧拉角，并绘制出相应图像
"""


import os
import threading

import cv2
import math
import numpy as np
from math import cos,sin

import tqdm
def get_best_chin(x,y):
    chin_x = x[8] * 0.7 + x[7] * 0.15 + x[9] * 0.15
    chin_y = y[8] * 0.7 + y[7] * 0.15 + y[9] * 0.15
    return (chin_x,chin_y)




model_points = np.array([
        (0.0, 0.0, 0.0),  # Nose tip
        (-165.0, 170.0, -135.0),  # Left eye left corner
        (165.0, 170.0, -135.0),  # Right eye right corne
        (-150.0, -150.0, -125.0),  # Left Mouth corner
        (150.0, -150.0, -125.0)  # Right mouth corner
    ])
def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size=20):
    """
    Prints the person's name and age.

    If the argument 'additional' is passed, then it is appended after the main info.

    Parameters
    ----------
    img : array
        Target image to be drawn on
    yaw : int
        yaw rotation
    pitch: int
        pitch rotation
    roll: int
        roll rotation
    tdx : int , optional
        shift on x axis
    tdy : int , optional
        shift on y axis

    Returns
    -------
    img : array
    """

    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1), int(y1)), (0, 0, 255), 2)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2), int(y2)), (0, 255, 0), 2)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3), int(y3)), (255, 0, 0), 2)

    return img
def pose_estimate(txt_path):
    with open(f'../train_lables/{txt_path}', 'r') as f:
        lines = f.readlines()
        img_name = txt_path.replace(".ttt",".jpg")
        img = cv2.imread(f'../coco_train/{img_name}')

        h,w = img.shape[0:2]
        focal_length = w / 2 / math.tan((60 / 2) * (math.pi / 180))
        center = (w // 2, h // 2)
        dist_coeffs = np.zeros((4, 1))
        camera_matrix = np.array([
            (focal_length,0,center[0]),
            (0,focal_length,center[1]),
            (0,0,1)
        ])
        for line in lines:
            data = line.split()
            points = data[6:]
            bbox = list(map(float,data[1:5]))
            bbox = list(map(int,bbox))
            # # Nose tip
            # (0.0, -330.0, -65.0),  # Chin
            # (-225.0, 170.0, -135.0),  # Left eye left corner
            # (225.0, 170.0, -135.0),  # Right eye right corne
            # (-150.0, -150.0, -125.0),  # Left Mouth corner
            # (150.0, -150.0, -125.0)  # Right mouth corner
            x = list(map(float, points[::3]))
            y = list(map(float, points[1::3]))

            left_eye = (x[36], y[36])
            right_eye = (x[45],y[45])
            nose = (x[30],y[30])

            left_mouse = (x[48], y[48])
            right_mouse = (x[54], y[54])

            image_points = np.array([
                nose,  # Nose tip
                left_eye,  # Left eye left corner
                right_eye,  # Right eye right corne

                left_mouse,  # Left Mouth corner
                right_mouse  # Right mouth corner
            ], dtype="double")
            (_, rvec, tvec) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs,
                                           flags=cv2.SOLVEPNP_UPNP)
            # Calculate Euler angles
            rmat = cv2.Rodrigues(rvec)[0]  # rotation matrix
            pmat = np.hstack((rmat, tvec))  # projection matrix
            eulers = cv2.decomposeProjectionMatrix(pmat)[-1]

            # Projecting a 3D point
            ## features
            x1 = bbox[0]
            y1 = bbox[1]

            x2 = bbox[0] + bbox[2]
            y2 = bbox[1] + bbox[3]
            for p in image_points:
                cv2.circle(img, (int(p[0]), int(p[1])), 1, (255, 125, 0), 1)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),1)
            pitch, yaw, roll = [math.radians(_) for _ in eulers]
            pitch = -math.degrees(math.asin(math.sin(pitch)))
            roll = -math.degrees(math.asin(math.sin(roll)))
            yaw = math.degrees(math.asin(math.sin(yaw)))
            score = 1 - abs(yaw) / 90 * 0.4 - abs(pitch) / 90  * 0.3 - abs(roll) / 90 * 0.3
            # print()
            # print("score:", score*100)
            # #
            # print('yaw:', yaw,
            #       'pitch:', pitch,
            #       'roll:', roll)
            draw_axis(img,yaw,pitch,roll,nose[0],nose[1])
            cv2.putText(img, str(int(score*100)),(int(nose[0]),int(nose[1])), 1, 1, (0, 0, 0), 1)
            # Display image
        # cv2.imshow("Output", img)
        # cv2.waitKey(0)
        cv2.imwrite(f'../6_points_img/{img_name}',img,[int(cv2.IMWRITE_JPEG_QUALITY),90])


def thread(txt_lst):
    for txt in tqdm.tqdm(txt_lst):
        pose_estimate(txt)
if __name__ == '__main__':
    # pose_estimate('000000002477.ttt')
    n_thread = 6
    txt_lst = os.listdir('../train_lables')
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
