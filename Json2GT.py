import json

# 从anchor开始，向下一共取7段，包含anchor,le_eye,r_eye,nose,l_mouse,r_mouse,chin
import os


def get_gt_data(item, index,flag):
    tmp = []
    if flag:
        for i in range(7):
            # 第一个是anchor
            if i == 0:
                x,y = item[index+i]['points'][0]
                w = item[index+i]['points'][1][0] - x
                h = item[index+i]['points'][1][1] - y
                tmp.append(x)
                tmp.append(y)
                tmp.append(w)
                tmp.append(h)
            else:
                x, y = item[index + i]['points'][0]
                tmp.append(x)
                tmp.append(y)
        return tmp
    # 如果是负样本，则只取候选框，其余全为-1
    else:
        x, y = item[index]['points'][0]
        w = item[index]['points'][1][0] - x
        h = item[index]['points'][1][1] - y
        tmp.append(x)
        tmp.append(y)
        tmp.append(w)
        tmp.append(h)
        tmp += [-1] * 12
        return tmp


if __name__ == '__main__':
    json_lst = os.listdir(r'D:\Data\COCO-WholeBody\back')
    for json_ in json_lst:
        print(json_)
        with open(f'D:/Data/COCO-WholeBody/back/{json_}', 'r') as f:
            js = json.load(f)
            n = len(js['shapes'])
            item = js['shapes']
            start = 0

            img_name = js['imagePath'].split('\\')[-1]

            # 保存到同名的txt中
            with open('../6_train_txt_center/' + img_name.replace('jpg','ttt'),'a') as f:

                while start < n:
                    tmp_str = []

                    # 目前只加负样本
                    if item[start]['label'] == 'anchor':
                        start += 7
                        continue
                        tmp_str += get_gt_data(item,start,flag=True)
                        start += 7
                    elif item[start]['label'] == 'novalid':
                        tmp_str += get_gt_data(item, start, flag=False)
                        start += 1
                    else:
                        raise("错误的格式，请检查")
                    tmp_str = ['%.3f' % x for x in tmp_str]
                    tmp_str.insert(0, 'bbox:')
                    tmp_str.insert(5, '6_ld')
                    tmp_str = list(map(str,tmp_str))
                    f.write(" ".join(tmp_str) + '\n')






