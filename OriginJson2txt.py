import collections
import json
from shutil import copyfile
import tqdm

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

with open('../coco_wholebody_val_v1.0.json', 'r') as f:
    data = json.load(f)
    for annotation in tqdm.tqdm(data['annotations']):
        flag = annotation["face_valid"]
        if flag:
            img_name = (12-count(annotation['image_id'])) * '0' + str(annotation['image_id']) + '.jpg'
            if img_name in tmp_data:
                tmp_data[img_name]['bboxes'].append(annotation['face_box'])
                tmp_data[img_name]['face_kpts_lst'].append(annotation['face_kpts'])
            else:
                tmp_data[img_name]['bboxes'] = [annotation['face_box']]
                tmp_data[img_name]['face_kpts_lst'] = [annotation['face_kpts']]

for img_name,data in tqdm.tqdm(tmp_data.items()):
    with open(f'../val_lables/{img_name.replace(".jpg",".ttt")}','w') as f:
        for i in range(len(data['bboxes'])):
            bbox = list(map(str,data['bboxes'][i]))
            face_kpts = list(map(str,data['face_kpts_lst'][i]))
            f.write("five_ld: " + " ".join(bbox) + " 68_ld: " + " ".join(face_kpts) + "\n")


