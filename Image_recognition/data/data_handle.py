import os
import cv2
from PIL import Image


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
                continue
            if image.shape[0] < 224 or image.shape[1] < 224:
                print(image.shape)
                os.remove(root + '/' + file)
                print(root + '/' + file, '删除成功')


# 图片 resize
def RGBResize(width=224, height=224):
    for root, dirs, files in os.walk('/home/tian/Desktop/image_增量/'):
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
def func(dirs='/home/tian/Desktop/image'):
    for root, dirs, files in os.walk(dirs):
        if len(files) < 300:
            print(root, '缺少', 300 - len(files))


# 文件重命名
def rename(dirs='/home/tian/Desktop/image/鼠标垫'):
    for i in os.listdir(dirs):
        os.rename(os.path.join(dirs, i), os.path.join(dirs, '1' + str(i)))


if __name__ == '__main__':
    remove()
    # rename()
    func()
