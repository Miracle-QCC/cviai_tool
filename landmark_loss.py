import os

import torch
import tqdm

from tools import get_iou
from tools import get_kpss_loss,IOUS

# root = r'B:\Data\retinaface_txt\retinaface_txt'
root = r'B:\Data\utils.1zip\utils\test_cviai\txt'

img_lst = []
gt = {}
with open(r'label.txt', 'r') as f:
    lines = f.readlines()
    img_name = None
    for line in lines:

        if "#" in line:
            img_name = line[1:].strip().split()[0]
            img_lst.append(img_name)
            gt[img_name] = []
        else:
            data = list(map(float,line.split()))
            gt[img_name].append(data)


loss = 0.0
TP = 0
FP = 0
total = 0
for img_name in tqdm.tqdm(img_lst):
    pre_txt_path = img_name.replace('jpg','txt')
    try:
        with open(os.path.join(root,pre_txt_path), 'r') as f:
            lines = f.readlines()[2:]
            if not lines:
                continue
            try:
                bboxes2 = torch.tensor(gt[img_name])[:, :4]

            except:
                print(f'gt为：{img_name}', gt[img_name])
                continue
            total += len(bboxes2)
            for line in lines:
                data = list(map(float, line.split()))
                # if data[4] < 0.5:
                #     FP += 1
                #     continue

                pre_box = data[:4]
                kpss = data[4:]
                pre_box = torch.tensor([pre_box])

                # 输出的bbox格式为 x,y,w,h,因此要转换为x1,y1,x2,y2
                # pre_box[0][2] = pre_box[0][2] + pre_box[0][0]
                # pre_box[0][3] = pre_box[0][3] + pre_box[0][1]
                ious = IOUS(pre_box, bboxes2)
                index = torch.argmax(ious)
                max_iou = torch.max(ious)
                if max_iou < 0.45:
                    FP += 1
                    continue
                TP += 1
                gt_kpss = gt[img_name][index][4:]
                W, H = gt[img_name][index][2] - gt[img_name][index][0], gt[img_name][index][3] - gt[img_name][index][1]

                # 如果该人脸没有kpss，则跳过
                if gt_kpss[0] == -1:
                    TP -= 1
                    continue
                delta = get_kpss_loss(kpss, gt_kpss, W, H)

                loss += delta

    except:
        pass


print("TP:",TP)
print("FP:",FP)
print("total:",total)
print('total_loss:', loss)
print("avg_loss:",loss / TP)




