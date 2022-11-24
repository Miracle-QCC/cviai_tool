import cv2
import tqdm
from math import cos,sin
import math
import numpy as np
from get_elur_angle import PoseEstimator,_radian2angle

# 获得人脸姿态分
def get_pose_score(bbox,kpt,img):
    W,H = bbox[2] - bbox[0],bbox[3] - bbox[1]
    left_eye = (kpt[0], kpt[1])
    right_eye = (kpt[2], kpt[3])
    nose = (kpt[4], kpt[5])
    left_mouth = (kpt[6], kpt[7])
    right_mouth = (kpt[8], kpt[9])
    l_max = min(left_eye[0],left_mouth[0])
    r_max = max(right_eye[0],right_mouth[0])

    eye_diff = math.sqrt((left_eye[0] - right_eye[0]) ** 2 + (left_eye[1] - right_eye[1]) ** 2)
    mouth_diff = math.sqrt((left_mouth[0] - right_mouth[0]) ** 2 + (left_mouth[1] - right_mouth[1]) ** 2)

    # 1.边界
    if left_eye[0] < bbox[0] or nose[0] < bbox[0] or left_mouth[0] < bbox[0] or \
            right_eye[0] > bbox[2] or nose[0] > bbox[2] or right_mouth[0] > bbox[2]:
        return 0.0
    # 2.眼睛和嘴巴太小，或者鼻子在最左最右
    elif nose[0] < l_max or nose[0] > r_max or eye_diff / W < 0.3 or \
            mouth_diff / W < 0.2:
        return 0.0

    else:
        points_5 = np.array([
            left_eye,
            right_eye,
            nose,
            left_mouth,
            right_mouth
        ])
        pose_es = PoseEstimator(img.shape)
        pose = pose_es.solve_pose_by_5_points(points_5)

        pitch, yaw, roll = pose_es.get_euler_angle(pose[0])
        pitch, yaw, roll = map(_radian2angle, [pitch, yaw, roll])
        return max(0, 1 - (abs(yaw) / 90 + abs(pitch) / 90 + abs(pitch) / 90) / 3.0)




def draw_anchor(img,anchor,color):
    # bbox的格式为 x1,y1,x2,y2
    bbox = anchor[:4]
    cv2.rectangle(img,(int(bbox[0]), int(bbox[1])),(int(bbox[2]), int(bbox[3])),color,1)

def draw_kpt(img,kpt):
    left_eye = (int(kpt[0]),int(kpt[1]))
    right_eye = (int(kpt[2]),int(kpt[3]))
    nose = (int(kpt[4]),int(kpt[5]))
    left_mouth = (int(kpt[6]),int(kpt[7]))
    right_mouth = (int(kpt[8]),int(kpt[9]))

    cv2.circle(img, left_eye, 1, (0, 255, 0), 1)
    cv2.circle(img, right_eye, 1, (0, 255, 0), 1)
    cv2.circle(img, nose, 1, (0, 255, 0), 1)
    cv2.circle(img, left_mouth, 1, (0, 255, 0), 1)
    cv2.circle(img, right_mouth, 1, (0, 255, 0), 1)


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


def draw_5(txtes):
    for txt_name in tqdm.tqdm(txtes):
        img_name = txt_name.replace('.ttt','.jpg')
        img = cv2.imread(f"coco_{state}/{img_name}")
        with open(f'{state}_lables/{txt_name}', 'r') as f:
            datas = f.readlines()
            for data in datas:
                data = data.split()
                x = data[6::3]
                y = data[7::3]

                x = list(map(float, x))
                y = list(map(float, y))

                left_eye = (int(sum(x[37:42]) / 5), int(sum(y[37:42]) / 5.0))
                right_eye = (int(sum(x[42:47]) / 5), int(sum(y[42:47]) / 5.0))

                nose = (int(x[30]), int(y[30]))

                left_mouse = (int(x[48]), int(y[48]))
                right_mouse = (int(x[54]), int(y[54]))

                cv2.circle(img, left_eye, 1, (0, 255, 0), 1)
                cv2.circle(img, right_eye, 1, (0, 255, 0), 1)
                cv2.circle(img, nose, 1, (0, 255, 0), 1)
                cv2.circle(img, left_mouse, 1, (0, 255, 0), 1)
                cv2.circle(img, right_mouse, 1, (0, 255, 0), 1)
            cv2.imwrite(f"processed_img/{img_name}", img)
