"""
从coco原始的json中获得anno.ttt
"""


import collections
import json
import math
import tqdm

def count_dis(p1,p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


# 计算iou
def iou(x1, y1, x2, y2, a1, b1, a2, b2):
    ax = max(x1, a1)  # 相交区域左上角横坐标
    ay = max(y1, b1)  # 相交区域左上角纵坐标
    bx = min(x2, a2)  # 相交区域右下角横坐标
    by = min(y2, b2)  # 相交区域右下角纵坐标

    area_N = (x2 - x1) * (y2 - y1)
    area_M = (a2 - a1) * (b2 - b1)

    w = max(0, bx - ax)
    h = max(0, by - ay)
    area_X = w * h

    try:
        return area_X / (area_N + area_M - area_X)
    except:
        return 0


# 过滤掉重复的和无效的bbox
def judge(boxes, face_kpts_lst):
    for face_kpt in face_kpts_lst:
        IOU = iou(boxes[0],boxes[1],boxes[0] + boxes[2],boxes[1] + boxes[3],face_kpt[0],face_kpt[1],face_kpt[0] + face_kpt[2],face_kpt[1] + face_kpt[3])
        if IOU > 0.65:
            return False
    return True

# 获得全0的bbox，看是遮挡还是重新标注
def judge_valid(boxes):
    if float(boxes[0]) != 0.0 and float(boxes[2]) != 0:
        return True

def count(n):
    if n < 10:
        return 1
    elif n < 100:
        return 2
    elif n < 1000:
        return 3
    elif n < 10000:
        return 4
    elif n < 100000:
        return 5
    else:
        return 6
tmp_data = collections.defaultdict(dict)

def thread(keys):
    if not tmp_data:
        return
    for img_name in tqdm.tqdm(keys):
        data = tmp_data[img_name]
        with open(f'../train/{img_name.replace(".jpg",".ttt")}','w') as f:
            for i in range(len(data['bboxes'])):
                bbox = list(map(str,data['bboxes'][i]))
                face_kpts = list(map(str,data['face_kpts_lst'][i]))
                f.write("bbox: " + " ".join(bbox) + " 68_ld: " + " ".join(face_kpts) + "\n")


if __name__ == '__main__':
    path = "train"
    COUNT = 0
    img_count = 0
    with open(f'../coco_wholebody_{path}_v1.0.json', 'r') as f:
    # with open(f'xxx.json', 'r') as f:
        data = json.load(f)
        for annotation in tqdm.tqdm(data['annotations']):
            flag = annotation["face_valid"]
            if flag:
                img_name = (12 - count(annotation['image_id'])) * '0' + str(annotation['image_id']) + '.jpg'

                if img_name in tmp_data and judge_valid(annotation['face_box']):
                    tmp_data[img_name]['bboxes'].append(annotation['face_box'])
                    tmp_data[img_name]['face_kpts_lst'].append(annotation['face_kpts'])
                    COUNT += 1
                elif judge_valid(annotation['face_box']):
                    tmp_data[img_name]['bboxes'] = [annotation['face_box']]
                    tmp_data[img_name]['face_kpts_lst'] = [annotation['face_kpts']]
                    COUNT += 1
                    img_count += 1

    len_n = len(tmp_data) // 1
    keys = list(tmp_data.keys())
    Processes = []
    thread(keys)
    print("总共多少张人脸：",COUNT)
    # for i in range(1):
    #     if i < 5:
    #         th = threading.Thread(target=thread,args=(keys[len_n*i:len_n*(i+1)],))
    #     else:
    #         th = threading.Thread(target=thread,args=(keys[len_n*(i+1):],))
    #     Processes.append(th)


    # for p in Processes:
    #     p.start()
    # for p in Processes:
    #     p.join()
    # with open('train_lables/000000000113.ttt','r') as f:
    #     lines = f.readlines()
    #     pts = list(map(float,lines[1].split()[6:]))
    #     face_pts = list(map(float,lines[2].split()[6:]))
    #     print(pts)
    # judge(pts,[face_pts])