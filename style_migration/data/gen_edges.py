import os
import numpy as np
from sklearn.model_selection import train_test_split
import random
import imutils
import cv2

def order_points(pts):
    # 四个点按照左上、右上、右下、左下
    # pts 是四个点的列表
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def four_point_transform(image, pts):  # image 为需要透视变换的图像  pts为四个点

    rect = order_points(pts)  # 四点排序
    (tl, tr, br, bl) = rect  # 方便计算

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
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped

def gen(path, flag):
    if flag == 'train':
        num = random.randint(1000000, 9999999)
        a_save_path = '/image/ai/machine-learn/style_migration/datasets/edges2plate/trainA/%s.jpg' % num
        num = random.randint(1000000, 9999999)
        b_save_path = '/image/ai/machine-learn/style_migration/datasets/edges2plate/trainB/%s.jpg' % num
    elif flag == 'test':
        num = random.randint(1000000, 9999999)
        a_save_path = '/image/ai/machine-learn/style_migration/datasets/edges2plate/testA/%s.jpg' % num
        num = random.randint(1000000, 9999999)
        b_save_path = '/image/ai/machine-learn/style_migration/datasets/edges2plate/testB/%s.jpg' % num
    image = cv2.imread(path)
    image = imutils.resize(image, height=256)
    cv2.imwrite(b_save_path, image)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 灰度图像
    gray = cv2.GaussianBlur(gray, (5, 5), 0)  # 高斯模糊
    edged = cv2.Canny(gray, 50, 200)  # 边缘检测
    dst = 255 - edged
    cv2.imwrite(a_save_path, dst)


# 获取目录下所有分类和图片数据
def get_classes(path='/home/tian/image/test/'):
    data_x_path = []
    # val = []  # 验证图片
    for i, file in enumerate(sorted(os.listdir(path))):
        data_x_path.append(path+file)
    test, train = train_test_split(data_x_path, shuffle=True, test_size=0.8)
    # test, val = train_test_split(test, shuffle=True, test_size=0.5)
    for i in train:
        gen(i, 'train')
    for i in test:
        gen(i, 'test')

get_classes()