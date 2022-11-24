import os.path
import cv2
import numpy as np
import tqdm
from draw_tool import draw_anchor,draw_kpt
from draw_tool import get_pose_score
def get_area(bbox):
    return (bbox[0] - bbox[2]) * (bbox[1] - bbox[3])
if __name__ == '__main__':

    imgs_root = r"B:\Data\benchmark\seq_10_27"
    pred_log_root = r"B:\Data\benchmark\seq_10_27\result.txt"
    dst_root = r"B:\Data\benchmark\seq_10_27\draw"
    # 从txt文件中读取出所有图片
    # with open(imgs_root, "r") as f:
    #     img_lst = f.readlines()

    avg_w_h = [0,0]
    count = 0
    #获得预测的结果
    with open(pred_log_root, "r") as f:
        # 每一张图片有三行结果
        # 第一行 ：/mnt/data/admin1_data/datasets/ivs_eval_set/image/seq_10_27/00000000.jpg
        # 第二行：boxes=[[462.917,600.313,499.896,660,0.6875],[709.167,92.5,971.667,423.333,0.617188],[708.906,610.938,738.021,645.625,0.59375]]
        # 第三行代表kpts ：[468.854,623.047,474.271,623.958,464.688,634.479,471.875,645,475.781,645.313],[739.167,214.167,821.458,199.583,744.167,263.255,765.417,337.5,832.813,322.5],[725.417,623.047,734.479,623.047,733.542,630.286,725.729,636.927,731.719,637.188]]
        lines = f.readlines()

        n = len(lines) // 3
        for i in tqdm.tqdm(range(n)):
            img_info = lines[i*3 + 0]
            name = img_info.split("/")[-1][:-1]
            boxes = None
            exec(lines[i*3+1])
            kpts = list(map(float,lines[i*3+2].replace("[","").replace("]","").split(",")))

            boxes = np.array(boxes).reshape(-1,5)
            kpts = np.array(kpts).reshape(-1,10)
            img_path = os.path.join(imgs_root,name)
            if len(boxes) < 3 or i in [57,61,63,64,65,66,67,68,69,70,71,72,121,122,123,125]:
                continue

            img = cv2.imread(img_path)


            idx = 0
            min_area = float('inf')
            for j,bbox in enumerate(boxes):
                area = get_area(bbox)

                if area < min_area:
                    min_area = area
                    idx = j
                # draw_anchor(img,bbox)
                # draw_kpt(img,kpt)
                # score = get_pose_score(bbox, kpt, img)
                # cv2.putText(img, str(int(bbox[4] * 100)), (int(bbox[0] + 2), int(bbox[1] + 2)), 1, 1, (0, 2, 255), 2)
                # cv2.putText(img, str(int(score * 100)), (int(bbox[2]), int(bbox[3])), 1, 1, (0, 2, 255), 2)
            draw_anchor(img,boxes[idx])
            draw_kpt(img,kpts[idx])

            print("第%s个"%i,boxes[idx][2] - boxes[idx][0],boxes[idx][3] - boxes[idx][1])
            count += 1
            avg_w_h[0] += boxes[idx][2] - boxes[idx][0]
            avg_w_h[1] += boxes[idx][3] - boxes[idx][1]
            # score = get_pose_score(bbox, kpt, img)
            if not os.path.exists(dst_root):
                os.mkdir(dst_root)

            cv2.imwrite(os.path.join(dst_root,name), img)
    print(avg_w_h)
    print(count)
    # bboxes = [[1110.42,66.6667,1278.33,369.167,0.695313]]
    # kpts = [[1142.92,168.958,1234.79,172.292,1170.83,225.208,1149.38,289.167,1218.23,291.667]]
    # boxes = np.array(bboxes).reshape(-1,5)
    # kpts = np.array(kpts).reshape(-1,10)
    # img_path = os.path.join(r'B:\Data\benchmark\seq_10_27\00000059.jpg')
    # img = cv2.imread(img_path)
    # # bbox = boxes[0]
    # # kpt = kpts[0]
    # # score = get_pose_score(bbox, kpt, img)
    # # cv2.putText(img, str(int(score * 100)), (int(bbox[0]), int(bbox[1])), 1, 1, (0, 255, 0), 2)
    # # cv2.imshow("x",img)
    # # [[1094.17,12.0833,1189.79,152.5,0.796875],[431.146,529.583,470.938,594.583,0.71875],[650.833,8.33333,913.333,341.667,0.59375],[680.833,542.917,710,577.083,0.539063]]
    # draw_anchor(img,[680.833,542.917,710,577.083,0.539063])
    # cv2.imshow("x",img)
    # cv2.waitKey(0)




