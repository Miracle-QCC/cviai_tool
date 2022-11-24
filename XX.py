import glob
import os
w = open("yolox_res.txt", 'w')
txts = glob.glob(r"B:\Data\COCO-WholeBody\utils\yolox_mask_res\xxx\yolox_mask_res\*txt")
for txt in txts:
    with open(txt, 'r') as f:

        lines = f. readlines()
        name = lines[0][:-1].split("/")
        txt_name = name[-2] + "/" + name[-1]

        data = lines[1][:-1]

        w.write("# " + txt_name + "\n")
        w.write(data + "\n")
