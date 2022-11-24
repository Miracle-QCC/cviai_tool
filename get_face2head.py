"""
计算人脸框与人头框的比值
"""

import os.path
import cv2
import numpy as np
import torch
import tqdm
from draw_tool import draw_anchor,draw_kpt
from draw_tool import get_pose_score
from tools import IOUS

def get_area(bbox):
    return (bbox[0] - bbox[2]) * (bbox[1] - bbox[3])


if __name__ == '__main__':

    imgs_root = r"B:\Data\benchmark\seq_10_27"
    pred_log_root = r"B:\Data\benchmark\seq_10_27\result.txt"
    head_prd_txt_root = r"B:\Data\benchmark\10-31\head_labels"
    dst_root = r"B:\Data\benchmark\seq_10_27\draw"
    # 从txt文件中读取出所有图片
    # with open(imgs_root, "r") as f:
    #     img_lst = f.readlines()

    avg_w_h = [0,0]
    count = 0
    #获得预测的结果
    with open(pred_log_root, "r") as f:
        # 每一张图片有三行结果
        # 第一行 ：/mnt/data/admin1_data/datasets/ivs_eval_set/image/seq_10_27/00000000.jpg
        # 第二行：boxes=[[462.917,600.313,499.896,660,0.6875],[709.167,92.5,971.667,423.333,0.617188],[708.906,610.938,738.021,645.625,0.59375]]
        # 第三行代表kpts ：[468.854,623.047,474.271,623.958,464.688,634.479,471.875,645,475.781,645.313],[739.167,214.167,821.458,199.583,744.167,263.255,765.417,337.5,832.813,322.5],[725.417,623.047,734.479,623.047,733.542,630.286,725.729,636.927,731.719,637.188]]
        lines = f.readlines()

        n = len(lines) // 3
        for i in tqdm.tqdm(range(n)):
            img_info = lines[i*3 + 0]
            name = img_info.split("/")[-1][:-1]
            boxes = None
            exec(lines[i*3+1])
            with open(os.path.join(head_prd_txt_root, name.replace("jpg", "txt")),'r') as h:
                heads = h.readlines()
            head_boxes = []
            for i,head in enumerate(heads):
                head_boxes.append(list(map(float,head.split()[1:])))

            head_boxes = np.array(head_boxes)

            # kpts = list(map(float,lines[i*3+2].replace("[","").replace("]","").split(",")))

            boxes = np.array(boxes).reshape(-1,5)
            # kpts = np.array(kpts).reshape(-1,10)
            img_path = os.path.join(imgs_root,name)

            ious = IOUS(torch.from_numpy(head_boxes),torch.from_numpy(boxes))
            img = cv2.imread(img_path)
            for i in range(head_boxes.shape[0]):
                iou_ts = ious[i]
                index = torch.argmax(iou_ts)
                max_iou = torch.max(iou_ts)
                # 如果当前人脸没有任何GT与之匹配，则是FP
                if max_iou < 0.2:
                    continue
                head = head_boxes[i]

                # 人头框是绿色的
                # 人脸框是蓝色的
                draw_anchor(img, (head[0], head[1], head[2], head[3]),(0,255,1))
                draw_anchor(img, boxes[index],(255,0,1))

                area_ratio = get_area(boxes[index]) / get_area(head)
                cv2.putText(img, ("head:face="+str(area_ratio)[:4]), (int(boxes[index][0] + 20), int(boxes[index][1] + 20)), 1, 1, (0, 0, 255), 1)


            # h,w = img.shape[:2]
            # head_x1, head_y1,head_x2,head_y2 = head_boxes[:,0] * w,head_boxes[:,1] * h,head_boxes[:,2] * w,head_boxes[:,3] * h


            # cv2.imshow("xxx",img)
            # cv2.waitKey(0)


            cv2.imwrite(os.path.join(dst_root,name), img)





