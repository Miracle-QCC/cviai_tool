import cv2
import torch
import numpy as np
from math import cos,sin




def get_area(bbox):
    return (bbox[0] - bbox[2]) * (bbox[1] - bbox[3])


def intersect(box_a, box_b):

    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:4].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:4].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


# 以tensor的形式计算iou
def IOUS(box_a, box_b):
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]

# 以tensor的形式计算iou,并且需要考虑GT是否有效
def IOUS2(box_a, box_b):
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]

def bbox_overlap(a, b):
    os = []
    for i in range(len(b)):
        _a = a[i]
        _b = b[i]
        x1 = np.maximum(_a[:,0], _b[0])
        y1 = np.maximum(_a[:,1], _b[1])
        x2 = np.minimum(_a[:,2], _b[2])
        y2 = np.minimum(_a[:,3], _b[3])
        w = x2-x1+1
        h = y2-y1+1
        inter = w*h
        aarea = (_a[:,2]-_a[:,0]+1) * (_a[:,3]-_a[:,1]+1)
        barea = (_b[2]-_b[0]+1) * (_b[3]-_b[1]+1)
        o = inter / (aarea+barea-inter)
        o[w<=0] = 0
        o[h<=0] = 0
        os.append(o)
    return os

def image_eval(pred, gt, iou_thresh):
    """ single image evaluation
    pred: Nx5
    gt: Nx4
    ignore:
    """

    _pred = pred.copy()
    _gt = gt.copy()
    pred_recall = np.zeros(_pred.shape[0])
    recall_list = np.zeros(_gt.shape[0])
    proposal_list = np.ones(_pred.shape[0])

    # _pred[:, 2] = _pred[:, 2] + _pred[:, 0]
    # _pred[:, 3] = _pred[:, 3] + _pred[:, 1]
    # _gt[:, 2] = _gt[:, 2] + _gt[:, 0]
    # _gt[:, 3] = _gt[:, 3] + _gt[:, 1]

    gt_overlap_list = bbox_overlap([_gt] * _pred.shape[0], [_pred[h] for h in range(_pred.shape[0])])

    for h in range(_pred.shape[0]):


        gt_overlap = gt_overlap_list[h]
        max_overlap, max_idx = gt_overlap.max(), gt_overlap.argmax()

        if max_overlap >= iou_thresh:
            if gt[max_idx][4] == -1:
                recall_list[max_idx] = -1
                proposal_list[h] = -1
            elif recall_list[max_idx] == 0:
                recall_list[max_idx] = 1

        r_keep_index = np.where(recall_list == 1)[0]
        pred_recall[h] = len(r_keep_index)

    return pred_recall, proposal_list


def img_pr_info(thresh_num, pred_info, proposal_list, pred_recall):
    pr_info = np.zeros((thresh_num, 2)).astype('float')
    fp = np.zeros((pred_info.shape[0],), dtype=np.int)
    # last_info = [-1, -1]
    for t in range(thresh_num):

        thresh = 1 - (t+1)/thresh_num
        r_index = np.where(pred_info[:, 4] >= thresh)[0]
        if len(r_index) == 0:
            pr_info[t, 0] = 0
            pr_info[t, 1] = 0
        else:
            r_index = r_index[-1]
            p_index = np.where(proposal_list[:r_index+1] == 1)[0]
            pr_info[t, 0] = len(p_index) #valid pred number
            pr_info[t, 1] = pred_recall[r_index] # valid gt number

            if t>0 and pr_info[t, 0] > pr_info[t-1,0] and pr_info[t, 1]==pr_info[t-1,1]:
                fp[r_index] = 1

    #print(pr_info[:10,0])
    #print(pr_info[:10,1])
    return pr_info, fp


def get_box(l_x, l_y, r_x, r_y, w,h, W,H):
    y = l_y * 0.5 + r_y * 0.5
    l_x = int(l_x)
    l_y = int(y - h // 10)
    r_x = int(r_x)
    r_y = int(y + h // 10)

    r_x = min(r_x, W)
    r_y = min(r_y, H)

    return l_x,l_y,r_x,r_y

def get_kpss_loss(pre, gt, W, H):
    x_gt = np.array(gt[::3]).reshape(5,1)
    y_gt = np.array(gt[1::3]).reshape(5,1)
    gt_xy = np.hstack((x_gt,y_gt))
    pre = np.array(pre).reshape(5,2)
    loss_ = pre - gt_xy
    loss_ = (loss_[:,0] / W) ** 2 + (loss_[:, 1] / H) ** 2
    return np.sqrt(loss_).sum() / 5.0


def getBestChin(nose, chin , bbox):
    if (float(nose[0]) - float(chin[0])) > 0 and abs(float(nose[0]) - float(chin[0])) / bbox[2] > 0.12:
        return -1
    elif (float(nose[0]) - float(chin[0])) < 0 and abs(float(nose[0]) - float(chin[0])) / bbox[2] > 0.12:
        return 1
    else:
        return 0


# 计算两个候选框的iou, 无相交时返回0
def get_iou(bbox1, bbox2):
    x1,y1,w1,h1 = bbox1
    a1,b1,w2,h2 = bbox2
    x2 = x1 + w1
    y2 = y1 + h1
    a2 = a1 + w2
    b2 = b1 + h2

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


def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size=20):
    """
    Prints the person's name and age.

    If the argument 'additional' is passed, then it is appended after the main info.

    Parameters
    ----------
    img : array
        Target image to be drawn on
    yaw : int
        yaw rotation
    pitch: int
        pitch rotation
    roll: int
        roll rotation
    tdx : int , optional
        shift on x axis
    tdy : int , optional
        shift on y axis

    Returns
    -------
    img : array
    """

    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1), int(y1)), (0, 0, 255), 2)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2), int(y2)), (0, 255, 0), 2)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3), int(y3)), (255, 0, 0), 2)

    return img