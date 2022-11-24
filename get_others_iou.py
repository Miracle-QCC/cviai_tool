import glob
import threading

import cv2
import  numpy as np
import torch
import tqdm

def get_area(bbox):
    return (bbox[0] - bbox[2]) * (bbox[1] - bbox[3])


def intersect(box_a, box_b):

    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


# 以tensor的形式计算iou
def IOUS(box_a, box_b):
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]

def get_box(l_x, l_y, r_x, r_y, w,h, W,H):
    y = l_y * 0.5 + r_y * 0.5
    l_x = int(l_x)
    l_y = int(y - h // 10)
    r_x = int(r_x)
    r_y = int(y + h // 10)

    r_x = min(r_x, W)
    r_y = min(r_y, H)

    return l_x,l_y,r_x,r_y




def thread(others_txts, flag):
    index = 0
    for txt in tqdm.tqdm(others_txts):
        img_path = txt.replace('ttt', 'jpg')
        with open(txt, 'r') as f:
            lines = f.readlines()
            img = cv2.imread(img_path)
            for line in lines:
                img = cv2.imread(img_path)
                H, W = img.shape[:2]
                data = list(map(float,line.split()))
                bbox = data[:4]
                bbox = [bbox[0] - 0.05*W, bbox[1] - 0.05*H, bbox[2] + 0.1 * W, bbox[3] + 0.1 * H]
                bbox = list(map(int,bbox))
                w, h = map(int, bbox[2:])
                count = 0
                # 每张人脸随机取5个区域
                while count < 5:
                    L_X = np.random.randint(bbox[0],(bbox[0] + bbox[2]))
                    L_Y = np.random.randint(bbox[1],(bbox[1] + bbox[3]))

                    R_X = np.random.randint(L_X, (bbox[0] + bbox[2]))
                    R_Y = np.random.randint(L_Y, (bbox[1] + bbox[3]))

                    ## 嘴巴
                    l_m_x = data[10]
                    l_m_y = data[11]
                    r_m_x = data[12]
                    r_m_y = data[13]

                    l_m_x, l_m_y, r_m_x, r_m_y = get_box(l_m_x, l_m_y, r_m_x, r_m_y,w,h,W,H)

                    ### 眼睛
                    l_e_x = data[4]
                    l_e_y = data[5]
                    r_e_x = data[6]
                    r_e_y = data[7]
                    l_e_x, l_e_y, r_e_x, r_e_y = get_box(l_e_x, l_e_y, r_e_x, r_e_y,w,h, W, H)

                    ### mask 鼻子
                    nose_x = data[8]
                    nose_y = data[9]

                    l_n_x = int(nose_x - w // 10)
                    r_n_x = int(nose_x + w // 10)
                    l_n_y = int(nose_y - h // 5)
                    r_n_y = int(nose_y + h // 10)


                    box = torch.from_numpy(np.array((L_X,L_Y, R_X,R_Y)).reshape(1,4))
                    bboxes = torch.from_numpy(np.array(([l_m_x, l_m_y, r_m_x, r_m_y],
                                       [l_e_x, l_e_y, r_e_x, r_e_y],
                                       [l_n_x, l_n_y, r_n_x, r_n_y])).reshape(3,4))

                    # 判断是否IOU通过且图片有一定大小
                    try:
                        if (IOUS(box,bboxes) < 0.01).all() and get_area((L_X,L_Y, R_X,R_Y)) > 100:
                            _img = img[L_Y:R_Y, L_X:R_X]
                            cv2.imwrite(f'../others/others_{flag}_{index}.jpg', _img)
                            count += 1
                            index += 1
                        else:
                            continue
                    except:
                        pass


if __name__ == '__main__':
    n_thread = 6
    root = r'../val_good_face/'
    others_txts = glob.glob(r'../val_good_face/*ttt')
    eve_len = len(others_txts) // n_thread
    Processes = []
    for i in range(n_thread + 1):
        if i < n_thread:
            p = threading.Thread(target=thread, args=(others_txts[eve_len*i:eve_len*(i+1)],i))
        else:
            p = threading.Thread(target=thread, args=(others_txts[eve_len * i:],i))

        Processes.append(p)

    for p in Processes:
        p.start()

    for p in Processes:
        p.join()
