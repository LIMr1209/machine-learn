import os
import cv2


# 数据清理 删除gif png 以及无效图片
def remove(dirs='/home/tian/Desktop/image'):
    for root, dirs, files in os.walk(dirs):
        for file in files:
            if file.endswith('.gif') or file.endswith('.png'):
                os.remove(root + '/' + file)
                print(root + '/' + file, '删除成功')
                continue
            image = cv2.imread(root + '/' + file)
            if image is None:
                os.remove(root + '/' + file)
                print(root + '/' + file, '删除成功')


if __name__ == '__main__':
    remove()
