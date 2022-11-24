"""
使用yolov7标注的人头框，为WIDERFACE的GT增加人头框

"""

import os.path

import torch
import tqdm
from tools import IOUS
import numpy as np
from multiprocessing import Process, Lock, Value

def strlist2np(lst):
    data = []
    for x in lst:
        data.append(list(map(float,x.split())))

    data = np.array(data)
    return data

###
# with open(r'B:\Data\retinaface\trian\labelv2.txt','r') as f:
#     lines = f.readlines()
#     end = len(lines)
#     start = 0
#     tmp_str = None
#     while start < end:
#         if "#" in lines[start]:
#             if tmp_str != None:
#                 if not os.path.exists(os.path.join(r'B:\Data\retinaface\trian\customize_txt', dirname)):
#                     os.mkdir(os.path.join(r'B:\Data\retinaface\trian\customize_txt', dirname))
#                 with open(os.path.join(os.path.join(r'B:\Data\retinaface\trian\customize_txt', dirname),img_name.replace("jpg",'txt')), 'w') as w:
#                     w.write(tmp_str)
#             data = lines[start][2:].split()
#             dirname = data[0].split('/')[0]
#             img_name = data[0].split('/')[1]
#             tmp_str = ""
#
#         else:
#             data = lines[start]
#             tmp_str += data
#
#         start += 1
head_box_root = r'B:\Data\add_head'
origin_gt_root = r'B:\Data\retinaface\trian\customize_txt'
dst_root = r'B:\Data\add_head_result'


def thread(head_dirs_lst,valid,no_valid):
    for dirname in tqdm.tqdm(head_dirs_lst):
        txt_lst = os.listdir(os.path.join(head_box_root, dirname))
        for txt in tqdm.tqdm(txt_lst):
            head_txt_path = os.path.join(os.path.join(head_box_root, dirname),txt)
            origin_txt_path = os.path.join(os.path.join(origin_gt_root,dirname),txt)
            result_path = os.path.join(os.path.join(dst_root,dirname),txt)
            if not os.path.exists(os.path.join(dst_root,dirname)):
                os.mkdir(os.path.join(dst_root,dirname))

            with open(head_txt_path, "r") as head:
                with open(origin_txt_path, "r") as origin:
                    heads = head.readlines()
                    origins = origin.readlines()

                    head_arr = torch.from_numpy(strlist2np(heads))
                    ori_arr = torch.from_numpy(strlist2np(origins))
            if ori_arr.shape[0] == 0:
                continue
            with open(result_path, 'w') as w:
                # print(txt)
                # print(ori_arr)
                # print(head_arr)
                ious = IOUS(ori_arr[:,:4], head_arr[:,1:])
                for i in range(ori_arr.shape[0]):
                    iou_ts = ious[i]
                    index = torch.argmax(iou_ts)
                    max_iou = torch.max(iou_ts)
                    # 如果当前人脸没有任何GT与之匹配，则是FP
                    if max_iou < 0.2:
                        no_valid.value += 1
                        origin_data = ori_arr[i].numpy()
                        w.write(" ".join(list(map(str,origin_data))) + " " + "-1 -1 -1 -1" + '\n')
                    else:

                        valid.value += 1
                        head_bbox = head_arr[index][1:].numpy()
                        origin_data = ori_arr[i].numpy()
                        w.write(" ".join(list(map(str,origin_data))) + " " + " ".join(list(map(str,head_bbox))) + '\n')
if __name__ == '__main__':
    head_dirs_lst = os.listdir(head_box_root)
    n_thread = 6
    n_len = len(head_dirs_lst) // n_thread
    P = []
    valid = Value("d", 0)
    no_valid = Value("d", 0)
    for i in range(n_thread + 1):
        if i < n_thread:
            pth = Process(target=thread, args=(head_dirs_lst[i*n_len:(i+1) * n_len], valid, no_valid))
        else:
            pth = Process(target=thread, args=(head_dirs_lst[i * n_len:], valid, no_valid))
        P.append(pth)

    for p in P:
        p.start()
    for p in P:
        p.join()
    print("有效的有:",valid.value)
    print("无效的有:",no_valid.value)



