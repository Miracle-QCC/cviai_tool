"""
从68个特征点提取出6个特征点
"""

import os
import tqdm


def thread_(txt_lst):
    for txt in tqdm.tqdm(txt_lst):
        with open(f'../val_lables/{txt}', 'r') as f:
            datas = f.readlines()
            with open(f'../6_val_txt/{txt}','w') as w:
                for data in datas:

                    data = data.split()
                    bbox = list(map(float, data[1:5]))
                    bbox = list(map(int, bbox))
                    tmp_lst = data[:6]
                    x = data[6::3]
                    y = data[7::3]

                    left_eye = [x[36], y[36]]
                    right_eye = [x[45], y[45]]
                    chin = [x[8], y[8]]
                    nose = [x[30], y[30]]

                    left_mouse = [x[48], y[48]]
                    right_mouse = [x[54], y[54]]
                    tmp_lst += left_eye
                    tmp_lst += right_eye
                    tmp_lst += nose
                    tmp_lst += left_mouse
                    tmp_lst += right_mouse
                    tmp_lst += chin
                    tmp_lst[5] = '6_ld'
                    w.write(" ".join(tmp_lst) + '\n')


if __name__ == '__main__':
    txt_lst = os.listdir('../val_lables')
    # n_thread = 6
    # every_len = len(txt_lst) // n_thread
    # Process = []
    # for i in range(n_thread + 1):
    thread_(txt_lst)