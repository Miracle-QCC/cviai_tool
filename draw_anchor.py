import cv2

# left_eye = (235, 282 )
# right_eye = (245, 279)
# nose = (240,285)
img = cv2.imread(r'B:\Data\retinaface\trian\images\9--Press_Conference\9_Press_Conference_Press_Conference_9_6.jpg')
with open(r'B:\Data\retinaface\trian\customize_txt\9--Press_Conference\9_Press_Conference_Press_Conference_9_6.txt', 'r') as f:
    lines = f.readlines()
    for str_lst in lines:
        str_lst = str_lst.split()


        boxes = list(map(float,str_lst))
        bbox = list(map(int,boxes[:4]))
        # left_eye = (int(boxes[4]), int(boxes[5]))
        # right_eye = (int(boxes[6]),int(boxes[7]))
        # nose = (int(boxes[8]),int(boxes[9]))
        # left_mouth = (int(boxes[10]),int(boxes[11]))
        # right_mouth = (int(boxes[12]),int(boxes[13]))
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 1), 1)

        # cv2.circle(img, left_eye, 1, (0, 255, 0), 1)
        # cv2.circle(img, right_eye, 1, (0, 255, 0), 1)
        # cv2.circle(img, nose, 1, (0, 255, 0), 1)
        # cv2.circle(img, left_mouth, 1, (0, 255, 0), 1)
        # cv2.circle(img, right_mouth, 1, (0, 255, 0), 1)
# bbox_str = "449.00000 330.00000 571.00000 479.00000".split()
# bbox = list(map(float,bbox_str))
# bbox = list(map(int,bbox))
# bbox = (233, 277,21+233,23+277 )

# img = cv2.imread(r'D:\Data\widerface\train\images\0--Parade\0_Parade_marchingband_1_205.jpg')
# cv2.rectangle(img,(bbox[0], bbox[1]),(bbox[2], bbox[3]),(0,255,1),1)
#
#
# cv2.circle(img,left_eye,1,(0,255,0),1)
# cv2.circle(img,right_eye,1,(0,255,0),1)
# cv2.circle(img,nose,1,(0,255,0),1)
# cv2.circle(img,left_mouth,1,(0,255,0),1)
# cv2.circle(img,right_mouth,1,(0,255,0),1)
# 109.732,600.411,128.001,621.304
# 809.41,190.884,820.298,203.93
# 306.468,239.821,330.628,267.884
#564.0 151.0 826.0 441.0
cv2.rectangle(img,(564,151),(826,441),(0,255,1),1)


cv2.imshow('x',img)
cv2.waitKey(0)
# cv2.imwrite('xxx.jpg',img)