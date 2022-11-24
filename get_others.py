import glob
import os
import cv2
import  numpy as np
import torch
import tqdm
from tools import get_iou

def get_area(bbox):
    return (bbox[0] - bbox[2]) * (bbox[1] - bbox[3])

np.random.seed(10)

index = 0
root = r'../val_good_face/'
eye_txts = glob.glob(r'../val_good_face/*ttt')
for txt in tqdm.tqdm(eye_txts):
    img_path = txt.replace('ttt', 'jpg')
    with open(txt, 'r') as f:
        lines = f.readlines()
        img = cv2.imread(img_path)
        for line in lines:
            img = cv2.imread(img_path)
            H, W = img.shape[:2]
            data = list(map(float,line.split()))
            bbox = data[:4]
            bbox = list(map(int,bbox))
            w, h = map(int, bbox[2:])

            with torch.no_grad():
                L_X = np.random.randint(bbox[0],(bbox[0] + bbox[2]))
                L_Y = np.random.randint(bbox[1],(bbox[1] + bbox[3]))

                R_X = np.random.randint(L_X, (bbox[0] + bbox[2]))
                R_Y = np.random.randint(L_Y, (bbox[1] + bbox[3]))

                ## 嘴巴
                l_m_x = data[10]
                l_m_y = data[11]
                r_m_x = data[12]
                r_m_y = data[13]

                y = l_m_y * 0.5 + r_m_y * 0.5
                l_m_x = int(l_m_x)
                l_m_y = int(y - h // 15)
                r_m_x = int(r_m_x)
                r_m_y = int(y + h // 12)


                r_m_x = min(r_m_x, W)
                r_m_y = min(r_m_y, H)

                ### 眼睛
                l_eye_x = data[4]
                l_eye_y = data[5]
                r_eye_x = data[6]
                r_eye_y = data[7]

                y = l_eye_y * 0.5 + r_eye_y * 0.5
                l_e_x = int(min(l_eye_x,l_m_x))
                l_e_y = int(y - h // 10)
                r_e_x = int(r_eye_x)
                r_e_y = int(y + h // 10)
                r_e_x = min(r_e_x, W)
                r_e_y = min(r_e_y, H)


                l_x = int(min(l_e_x,l_m_x))

                ### mask 鼻子
                nose_x = data[8]
                nose_y = data[9]

                l_n_x = int(nose_x - w // 10)
                r_n_x = int(nose_x + w // 10)
                l_n_y = int(nose_y - h // 10)
                r_n_y = int(nose_y + h // 10)

                ### 左脸
                # _img = img[r_e_y:(bbox[1]+bbox[3]), bbox[0]:l_m_x]
                try:
                    if get_area((bbox[0],bbox[1], l_x,bbox[1] + bbox[3])) > get_area((bbox[0],r_e_y, l_m_x,bbox[1] + bbox[3])):
                        _img = img[bbox[1]:(bbox[1] + bbox[3]), bbox[0]:l_x]
                    else:
                        _img = img[r_e_y:(bbox[1] + bbox[3]),bbox[0]:l_m_x]
                    cv2.imwrite(f'B:/Data\COCO-WholeBody/3cls/val/data/others_{index}.jpg',_img)
                    index += 1
                except:
                    print("图片尺寸不对")


                r_f_x = max(r_n_x,r_m_x)
                ### 右脸
                try:
                    if get_area((r_e_x,r_m_y, bbox[0]+bbox[2],bbox[1] + bbox[3])) > get_area((r_f_x,r_e_y, bbox[0]+bbox[2],bbox[1] + bbox[3])):
                        __img = img[r_m_y:(bbox[1] + bbox[3]),r_e_x:(bbox[0]+bbox[2])]
                    else:
                        __img = img[r_e_y:(bbox[1] + bbox[3]),r_f_x:(bbox[0]+bbox[2])]
                    cv2.imwrite(f'B:/Data\COCO-WholeBody/3cls/val/data/others_{index}.jpg', __img)
                    index += 1
                except:
                    print("图片尺寸不对")

                ## 嘴巴以下
                try:
                    ___img = img[r_m_y:(bbox[1] + bbox[3]),bbox[0]:(bbox[0] + bbox[2])]
                    cv2.imwrite('xxx.jpg', ___img)
                    cv2.imwrite(f'B:/Data\COCO-WholeBody/3cls/val/data/others_{index}.jpg', ___img)
                    index += 1
                except:
                    print("图片尺寸不对")


