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
            bbox = [int(x) for x in bbox]
            w,h = map(int,bbox[2:])
            l_m_x = data[10]
            l_m_y = data[11]
            r_m_x = data[12]
            r_m_y = data[13]

            y = l_m_y *0.5 + r_m_y * 0.5
            l_x = int(l_m_x - w // 20)
            l_y = int(y - h // 9)
            r_x = int(r_m_x + w // 20)
            r_y = int(y + h // 9)
            img = cv2.imread(img_path)
            h,w = img.shape[:2]
            r_x = min(r_x, w)
            r_y = min(r_y,h)
            try:
                _img = mask_other(img, l_x,l_y,r_x,r_y)
                # _img = mask_other(img, bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1]  + bbox[3])
                # cv2.imshow('x', _img)

                # cv2.waitKey(0)

                cv2.imwrite(f'B:/Data\COCO-WholeBody/3cls/val/data/mouth_{index}.jpg',_img)
                index += 1
            except:
                print('嘴巴太小，无法输出')

