import glob
import os
import cv2
import  numpy as np
import tqdm


def mask_other(img, l_x,l_y, r_x,r_y):

    _img = np.zeros((r_y - l_y, r_x - l_x, 3))
    for j in range(l_x,r_x):
        for i in range(l_y,r_y):
            _img[i-l_y][j-l_x] = img[i][j]
    return _img

index = 0
root = r'../val_good_face/'
eye_txts = glob.glob(r'../val_good_face/*ttt')
for txt in tqdm.tqdm(eye_txts):
    img_path = txt.replace('ttt', 'jpg')
    with open(txt, 'r') as f:
        lines = f.readlines()
        for line in lines:
            data = list(map(float,line.split()))
            bbox = data[:4]
            w,h = map(int,bbox[2:])
            l_eye_x = data[4]
            l_eye_y = data[5]
            r_eye_x = data[6]
            r_eye_y = data[7]

            # 框高的中心基于左右两边的纵坐标计算
            y = l_eye_y *0.5 + r_eye_y * 0.5

            # 绘制眼睛框， 宽标稍微扩一点，框的高要根据候选框选择
            l_x = int(l_eye_x - w // 20)
            l_y = int(y - h // 9)
            r_x = int(r_eye_x + w // 20)
            r_y = int(y + h // 9)
            img = cv2.imread(img_path)
            h,w = img.shape[:2]
            r_x = min(r_x, w)
            r_y = min(r_y,h)
            try:

                _img = mask_other(img, l_x,l_y,r_x,r_y)

                cv2.imwrite(f'B:/Data\COCO-WholeBody/3cls/val/data/eye_{index}.jpg',_img)
                index += 1
            except:
                print('眼睛太小，无法输出')

