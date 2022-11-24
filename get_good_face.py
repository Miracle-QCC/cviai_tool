import os
import shutil

import tqdm
txt_root = '../retinaface_val_labels'
txt_lst = os.listdir(txt_root)
index = 1
for txt in tqdm.tqdm(txt_lst):
    with open(os.path.join(txt_root, txt), 'r') as f:
        lines = f.readlines()
        for line in lines:
            data = list(map(float, line.split()))
            if data[-1] > 0.75:
                with open(f'../val_good_face/{index}.ttt', 'w') as w:
                    tmp_str = [str(x) for x in data]
                    w.write(" ".join(tmp_str))
                    shutil.copy(f'../coco_val/{txt.replace("ttt","jpg")}', f'../val_good_face/{index}.jpg')
                    index += 1

