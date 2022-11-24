"""
数据的目录结构为：
3cls:
    - train
        -- data   # 存放imgs
        anno.ttt  # 标签txt，格式为  xxx.img 0
    - val
        -- data  # 窜访imgs
        anno.ttt # 同上
"""

import os

# 0  eye
# 1 mouth
# 2 other

root = '../3cls/val/data'
img_lst = os.listdir(root)
with open('../3cls/val/anno.ttt', 'w') as f:
    for img_name in img_lst:
        if 'eye' in img_name:
            f.write(img_name + " 0" + '\n')
        elif 'mouth' in img_name:
            f.write(img_name + " 1" + '\n')
        else:
            f.write(img_name + " 2" + '\n')


