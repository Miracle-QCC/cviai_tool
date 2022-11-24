import numpy as np
from PIL import Image
import cv2
def count_var(img):
    mean = np.mean(img)
    img_ = img - mean
    return np.mean(img_ ** 2)

# imgfile = r"B:\Data\benchmark\ipc_test_face\capture_face_139_0.910112.png"
imgfile = r'face_3_out.png'
frame = cv2.imread(imgfile)
img2gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
lap_img = cv2.Laplacian(img2gray, cv2.CV_64F)
var = lap_img.var()
std = lap_img.std()

print(std)
# OriginalPic = np.array(Image.open(imgfile).convert('L'), dtype=np.uint8)
# img = np.zeros((OriginalPic.shape[0]+2, OriginalPic.shape[1]+2), np.uint8)
# #########  制造遍历图像  ###################
# for i in range(1, img.shape[0]-1):
#     for j in range(1, img.shape[1]-1):
#         img[i][j] = OriginalPic[i-1][j-1]
# score = 0.0
# LaplacePic = np.zeros((OriginalPic.shape[0], OriginalPic.shape[1]), dtype=np.uint8)
# kernel = [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
# # for i in range(0, LaplacePic.shape[0]):
# #     for j in range(0, LaplacePic.shape[1]):
# #         LaplacePic[i][j] = np.sum(np.multiply(kernel, img[i:i+3, j:j+3]))
# #         score += LaplacePic[i][j]
#
#
# cout_var = count_var(LaplacePic)
#
#
#
# score /= 1.2 * img.shape[0] * img.shape[1] * 1.2 * 20
# print(score)
# cv2.imshow("Original", OriginalPic)
# cv2.imshow("Laplace", LaplacePic)

# cv2.imshow("opencv", lap_img)
# cv2.waitKey(0)
