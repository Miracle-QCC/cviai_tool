"""
获得srcfd模型需要的数据格式
结构为：
/data/coco:
    train
        -images:
            xxx.jpg
            yy.jpg
        append.ttt # 存储gt，格式为
            # xxx.jpg W H     ###其中W H为图片的宽和高
            bbox的四个点(x1,y1,x2,y2) (x,y,number) * 5  pose_score # 其中number为0/1，用于隔开每个landmark点,pose_score是新增的
    val
        -images:
            xxx.jpg
            yy.jpg
        append.ttt # 存储gt，格式为
            # xxx.jpg W H     ###其中W H为图片的宽和高
            bbox的四个点(x,y,w,h) (x,y,number) * 5  # 其中number为0/1，用于隔开每个landmark点
"""
import os
import cv2
import tqdm

img_root = '../coco_train'
txt_root = "../6_lables_txt_center"
txts_lst = os.listdir(txt_root)
def conver2ScrfdBox(data):
    data[2] = str(float(data[0]) + float(data[2]))
    data[3] = str(float(data[1]) + float(data[3]))
    return data

with open('widerface_label.txt', 'w') as f:
    for txt in tqdm.tqdm(txts_lst):
        img_name = txt.replace('ttt', 'jpg')
        img_path = os.path.join(img_root, img_name)
        txt_path = os.path.join(txt_root, txt)
        img = cv2.imread(img_path)
        H,W = img.shape[:2]

        # 先写入 # xx.jpg W H
        f.write(f"# cocofaces/{img_name} {W} {H}\n")

        with open(txt_path, 'r') as r:
            lines = r.readlines()
            for line in lines:
                data = line.split()
                # 转换为标签
                data = conver2ScrfdBox(data)
                tmp_line = data[:4] + data[4:6] + ['0'] + data[6:8] + ['0'] + data[8:10] + ['0'] + data[10:12] + ['0'] + data[12:14] + ['0']
                # print(tmp_line)
                f.write(" ".join(tmp_line) + '\n')




