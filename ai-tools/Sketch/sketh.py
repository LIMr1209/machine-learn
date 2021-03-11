import argparse
import cv2
import imutils

import numpy as np
import cv2

def order_points(pts):
    #四个点按照左上、右上、右下、左下

    #pts 是四个点的列表
    rect = np.zeros((4,2),dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts,axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def four_point_transform(image, pts):  #image 为需要透视变换的图像  pts为四个点

    rect = order_points(pts)  #四点排序
    (tl, tr, br, bl) = rect   #方便计算

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))


    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped

# ap = argparse.ArgumentParser()
# ap.add_argument('-i','--image',required=True,help="Path to image file")
#
# args = vars(ap.parse_args())

#flag: 获取了图像路径

image = cv2.imread('test.jpg')
image = imutils.resize(image,height=500)

gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) #灰度图像
# gray = cv2.GaussianBlur(gray,(5,5),0)          #高斯模糊
edged = cv2.Canny(gray,75,200)                  #边缘检测



# flag : Test1 = BLOCK
print("STEP 1 Edge Detection")
cv2.imshow("Edge",edged)
cv2.waitKey(0)
cv2.destroyAllWindows()