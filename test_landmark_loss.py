"""
用于测试SCRFD模型输出的landmark与真实数据的误差，误差会根据候选框进行归一化
"""

import os.path
import sys

import torch
import cv2
import numpy as np
import tqdm
from scrfd import SCRFD
from tools import get_kpss_loss,IOUS

img_lst = []
gt = {}


loss = 0.0
TP = 0
FP = 0
kpss_count = 0
if __name__ == '__main__':
    # root表示imgs的地址
    root = sys.argv[1]
    # total_face: 76006   widerface训练集中所有的带有landmark的人脸总数
    total_face = 76006
    # 读取标签数据
    with open(r'label.txt', 'r') as f:
        lines = f.readlines()
        img_name = None
        for line in lines:
            if "#" in line:
                img_name = line[1:].strip().split()[0]
                img_lst.append(img_name)
                gt[img_name] = []
            else:
                data = list(map(float, line.split()))
                gt[img_name].append(data)

    for img_name in tqdm.tqdm(img_lst):

        # 在root下找到与GT同名的图片
        img = cv2.imread(os.path.join(root, img_name.replace('/', '\\')))

        # scrfd
        detector = SCRFD(model_file=r'scrfd_500m_bnkps_640_640.onnx')
        bboxes, kpss = detector.detect(img, 0.5, input_size=(640, 640))

        try:
            bboxes2 = torch.tensor(gt[img_name])[:,:4]

        except:
            print(f'gt为：{img_name}',gt[img_name])
            continue

        bboxes = torch.from_numpy(bboxes)
        ious = IOUS(bboxes[:,:4], bboxes2)
        for i in range(kpss.shape[0]):
            iou_ts = ious[i]
            index = torch.argmax(iou_ts)
            max_iou = torch.max(iou_ts)
            # 如果当前人脸没有任何GT与之匹配，则是FP
            if max_iou < 0.45:
                FP += 1
                continue
            #  正样本 + 1
            TP += 1
            gt_kpss = gt[img_name][index][4:]
            W, H = gt[img_name][index][2] - gt[img_name][index][0], gt[img_name][index][3] - gt[img_name][index][1]

            # 如果该人脸没有kpss，则跳过，并且TP - 1
            if gt_kpss[0] == -1:
                TP -= 1
                continue
            # 计算五个点的平均loss
            delta = get_kpss_loss(kpss[i], gt_kpss, W, H)
            loss += delta
    recall = TP / total_face
    precession = TP / (TP + FP)
    F1 = recall * precession * 2 /(precession + recall)

    print("TP:",TP)
    print("FP:",FP)
    print('total_loss:', loss)
    print("avg_loss:",loss / TP)
    print("f1_score:",F1)



