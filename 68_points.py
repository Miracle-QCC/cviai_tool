# -*- coding: utf-8 -*-
"""
根据coco_wholebody数据集的GTjson文件获得人脸的六个特征点，然后利用这些特征点来计算欧拉角，并绘制出相应图像；
但实际测试结果来看，并不是很准
"""
import os
import threading

import cv2
import math
import numpy as np
from math import cos,sin

import tqdm

model_points = np.array([
            [-73.393523, -29.801432, -47.667532],
            [-72.775014, -10.949766, -45.909403],
            [-70.533638,   7.929818, -44.84258 ],
            [-66.850058,  26.07428 , -43.141114],
            [-59.790187,  42.56439 , -38.635298],
            [-48.368973,  56.48108 , -30.750622],
            [-34.121101,  67.246992, -18.456453],
            [-17.875411,  75.056892,  -3.609035],
            [  0.098749,  77.061286,   0.881698],
            [ 17.477031,  74.758448,  -5.181201],
            [ 32.648966,  66.929021, -19.176563],
            [ 46.372358,  56.311389, -30.77057 ],
            [ 57.34348 ,  42.419126, -37.628629],
            [ 64.388482,  25.45588 , -40.886309],
            [ 68.212038,   6.990805, -42.281449],
            [ 70.486405, -11.666193, -44.142567],
            [ 71.375822, -30.365191, -47.140426],
            [-61.119406, -49.361602, -14.254422],
            [-51.287588, -58.769795,  -7.268147],
            [-37.8048  , -61.996155,  -0.442051],
            [-24.022754, -61.033399,   6.606501],
            [-11.635713, -56.686759,  11.967398],
            [ 12.056636, -57.391033,  12.051204],
            [ 25.106256, -61.902186,   7.315098],
            [ 38.338588, -62.777713,   1.022953],
            [ 51.191007, -59.302347,  -5.349435],
            [ 60.053851, -50.190255, -11.615746],
            [  0.65394 , -42.19379 ,  13.380835],
            [  0.804809, -30.993721,  21.150853],
            [  0.992204, -19.944596,  29.284036],
            [  1.226783,  -8.414541,  36.94806 ],
            [-14.772472,   2.598255,  20.132003],
            [ -7.180239,   4.751589,  23.536684],
            [  0.55592 ,   6.5629  ,  25.944448],
            [  8.272499,   4.661005,  23.695741],
            [ 15.214351,   2.643046,  20.858157],
            [-46.04729 , -37.471411,  -7.037989],
            [-37.674688, -42.73051 ,  -3.021217],
            [-27.883856, -42.711517,  -1.353629],
            [-19.648268, -36.754742,   0.111088],
            [-28.272965, -35.134493,   0.147273],
            [-38.082418, -34.919043,  -1.476612],
            [ 19.265868, -37.032306,   0.665746],
            [ 27.894191, -43.342445,  -0.24766 ],
            [ 37.437529, -43.110822,  -1.696435],
            [ 45.170805, -38.086515,  -4.894163],
            [ 38.196454, -35.532024,  -0.282961],
            [ 28.764989, -35.484289,   1.172675],
            [-28.916267,  28.612716,   2.24031 ],
            [-17.533194,  22.172187,  15.934335],
            [ -6.68459 ,  19.029051,  22.611355],
            [  0.381001,  20.721118,  23.748437],
            [  8.375443,  19.03546 ,  22.721995],
            [ 18.876618,  22.394109,  15.610679],
            [ 28.794412,  28.079924,   3.217393],
            [ 19.057574,  36.298248,  14.987997],
            [  8.956375,  39.634575,  22.554245],
            [  0.381549,  40.395647,  23.591626],
            [ -7.428895,  39.836405,  22.406106],
            [-18.160634,  36.677899,  15.121907],
            [-24.37749 ,  28.677771,   4.785684],
            [ -6.897633,  25.475976,  20.893742],
            [  0.340663,  26.014269,  22.220479],
            [  8.444722,  25.326198,  21.02552 ],
            [ 24.474473,  28.323008,   5.712776],
            [  8.449166,  30.596216,  20.671489],
            [  0.205322,  31.408738,  21.90367 ],
            [ -7.198266,  30.844876,  20.328022]
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
        img_name = txt_path.replace(".ttt", ".jpg")
        img = cv2.imread(f'../coco_train/{img_name}')

        h, w = img.shape[0:2]
        focal_length = w / 2 / math.tan((60 / 2) * (math.pi / 180))
        center = (w // 2, h // 2)
        dist_coeffs = np.zeros((4, 1))
        camera_matrix = np.array([
            (focal_length, 0, center[0]),
            (0, focal_length, center[1]),
            (0, 0, 1)
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
        h, w = bbox[2:]
        focal_length = w / 2 / math.tan((60 / 2) * (math.pi / 180))
        center = (w // 2, h // 2)
        dist_coeffs = np.zeros((4, 1))
        camera_matrix = np.array([
            (focal_length, 0, center[0]),
            (0, focal_length, center[1]),
            (0, 0, 1)
        ])
        # left_eye = (x[36], y[36])
        # right_eye = (x[45],y[45])
        # chin = (x[8],y[8])
        nose = (x[30], y[30])
        #
        # left_mouse = (x[48], y[48])
        # right_mouse = (x[54], y[54])
        xy = [(x_,y_) for x_,y_ in zip(x,y)]
        image_points = np.array([
            xy
        ], dtype="double")
        (_, rvec, tvec) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs,
                                       flags=cv2.SOLVEPNP_UPNP)
        # Calculate Euler angles
        rmat = cv2.Rodrigues(rvec)[0]  # rotation matrix
        pmat = np.hstack((rmat, tvec))  # projection matrix
        eulers = cv2.decomposeProjectionMatrix(pmat)[-1]

        for points in xy:
            cv2.circle(img, (int(points[0]), int(points[1])), 1, (255, 125, 0), 1)
        # cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),1)
        pitch, yaw, roll = [math.radians(_) for _ in eulers]
        pitch = -math.degrees(math.asin(math.sin(pitch)))
        roll = -math.degrees(math.asin(math.sin(roll)))
        yaw = math.degrees(math.asin(math.sin(yaw)))
        score = 1 - abs(yaw) / 90 * 0.4 - abs(pitch) / 90 * 0.3 - abs(roll) / 90 * 0.3
        # print()
        # print("score:", score*100)
        # #
        # print('yaw:', yaw,
        #       'pitch:', pitch,
        #       'roll:', roll)
        draw_axis(img, yaw, pitch, roll, nose[0], nose[1])
        cv2.putText(img, str(int(score * 100)), (int(nose[0]), int(nose[1])), 1, 1, (0, 0, 0), 1)
        # Display image
        # cv2.imshow("Output", img)
        # cv2.waitKey(0)
    cv2.imwrite(f'../6_points_img/{img_name}', img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

def thread(txt_lst):
    for txt in tqdm.tqdm(txt_lst):
        pose_estimate(txt)

if __name__ == '__main__':
    # pose_estimate('000000002477.ttt')
    n_thread = 6
    txt_lst = os.listdir('../train_lables')
    len_evey = len(txt_lst) // n_thread
    Processes = []
    for i in range(6 + 1):
        if i < 6:
            p = threading.Thread(target=thread, args=(txt_lst[i * len_evey:(i + 1) * len_evey],))
        else:
            p = threading.Thread(target=thread, args=(txt_lst[i * len_evey:],))
        Processes.append(p)

    for p in Processes:
        p.start()
    for p in Processes:
        p.join()
