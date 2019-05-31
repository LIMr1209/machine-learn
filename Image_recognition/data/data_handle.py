import os, shutil, time
import random
import cv2
from PIL import Image


# 数据清理 删除gif png 以及无效图片
def remove(dir='/home/tian/Desktop/image_new'):
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith('.gif') or file.endswith('.png'):
                os.remove(root + '/' + file)
                print(root + '/' + file, '删除成功')
                continue
            image = cv2.imread(root + '/' + file)
            if image is None:
                os.remove(root + '/' + file)
                print(root + '/' + file, '删除成功')
                continue
            if image.shape[0] < 224 or image.shape[1] < 224:
                print(image.shape)
                os.remove(root + '/' + file)
                print(root + '/' + file, '删除成功')


# 图片 resize
def RGBResize(dir='/home/tian/Desktop/image_增量/', width=224, height=224):
    for root, dirs, files in os.walk(dir):
        for file in files:
            img = Image.open(root + '/' + file)
            if img.mode == 'RGBA' or img.mode == 'P':
                img = img.convert('RGB')
            try:
                root_resize = root.replace('image_增量', 'image_resize')
                if not os.path.exists(root_resize):
                    os.makedirs(root_resize)
                new_img = img.resize((width, height), Image.ANTIALIAS)
                new_img.save(root_resize + '/' + file)
            except Exception as e:
                print(e)


# 图片统计
def image_stat(dir='/home/tian/Desktop/image_test'):
    for root, dirs, files in os.walk(dir):
        if len(files) < 300:
            print(root, '缺少', 300 - len(files))


# opalus 下载图片添加至训练集重命名
def opalus_err(dir='/home/tian/Desktop/image_train/'):
    for root, dirs, files in os.walk(dir):
        for file in files:
            tag = root[root.rfind('/') + 1:]
            # 移动
            shutil.move(os.path.join(root, file), os.path.join('/image/image/' + tag, file))
            a = int(time.time())
            b = random.randint(10, 100)
            # 重命名
            new_file = str(a) + str(b) + '.jpg'
            os.rename(os.path.join('/image/image/' + tag, file), os.path.join('/image/image/' + tag, new_file))


if __name__ == '__main__':
    # remove()
    # rename()
    # image_stat()
    opalus_err()
