import os

import tqdm

from tools import get_iou

gt_root = '/dataset/retinaface_train_labels'
pred_root = './customize_evaluate/customize_train_txt'
gt_lst = os.listdir(gt_root)
pred_lst = os.listdir(pred_root)

avg_score = 0.0
face_num = 0
f = open('lease.ttt','w')
for gt in tqdm.tqdm(gt_lst):
    with open(os.path.join(gt_root, gt),'r') as f_g:
        try:
            with open(os.path.join(pred_root,gt)) as f_p:
                g_lines = f_g.readlines()
                p_lines = f_p.readlines()
                for g_line in g_lines:
                    g_data = g_line.split()
                    g_data = list(map(float,g_data))
                    g_box = g_data[:4]
                    # g_box[2] = g_box[2] - g_box[0]
                    # g_box[3] = g_box[3] - g_box[1]
                    index = -1
                    max_iou = 0.1
                    face_num += 1
                    for i,p_line in enumerate(p_lines[2:]):
                        p_data = p_line.split()
                        p_data = list(map(float,p_data))
                        p_box = p_data[:4]
                        iou = get_iou(g_box,p_box)
                        if iou > max_iou:
                            index = i
                    if index == -1:
                        avg_score += g_data[-1]
                    else:
                        avg_score += abs(g_data[-1] - p_data[-1])
        except:
            print("丢失的有:",gt)
            f.write(gt)
            f.write('\n')

f.close()

print(avg_score)
print(face_num)
print(avg_score / face_num)




