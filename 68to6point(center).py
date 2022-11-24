"""
从68个特征点提取出6个特征点,其中眼睛的特征为眼睛中心，而不是眼角
"""

import os
import threading

import tqdm


def thread_(txt_lst):
    for txt in tqdm.tqdm(txt_lst):
        with open(f'../train_lables/{txt}', 'r') as f:
            datas = f.readlines()
            with open(f'../6_train_txt_center/{txt}','w') as w:
                for data in datas:

                    data = data.split()
                    bbox = list(map(float, data[1:5]))
                    bbox = list(map(int, bbox))
                    tmp_lst = data[:6]
                    x = data[6::3]
                    y = data[7::3]

                    # 对6个点的坐标求平均获得眼睛中兴坐标
                    left_eye = [sum(map(float,x[36:(36+6)])) / 6.0, sum(map(float,y[36:(36+6)])) / 6.0]
                    right_eye = [sum(map(float, x[42:(42+6)])) / 6.0, sum(map(float, y[42:(42+6)])) / 6.0]
                    chin = [x[8], y[8]]
                    nose = [x[30], y[30]]

                    left_mouse = [x[48], y[48]]
                    right_mouse = [x[54], y[54]]
                    tmp_lst += list(map(str,left_eye))
                    tmp_lst += list(map(str,right_eye))
                    tmp_lst += nose
                    tmp_lst += left_mouse
                    tmp_lst += right_mouse
                    tmp_lst += chin
                    tmp_lst[5] = '6_ld'
                    w.write(" ".join(tmp_lst) + '\n')


if __name__ == '__main__':
    txt_lst = os.listdir('../train_lables')
    n_thread = 6
    every_len = len(txt_lst) // n_thread
    Process = []
    for i in range(n_thread + 1):
        if i < n_thread:
            p = threading.Thread(target=thread_, args=(txt_lst[i*every_len:(i+1)*every_len],))
        else:
            p = threading.Thread(target=thread_, args=(txt_lst[i * every_len:],))
        Process.append(p)


    for p in Process:
        p.start()

    for p in Process:
        p.join()

    # thread_(txt_lst)