import multiprocessing

import os

from PIL import Image


def gen_img(path, new_path):
    # 图像组成：红绿蓝  （RGB）三原色组成    亮度（255,255,255）
    img = Image.open(path)
    new = Image.new("L", img.size, 255)
    width, height = img.size
    img = img.convert("L")
    # print(img.size)
    # print(img.mode) #RBG
    #
    # img_get = img.getpixel((0, 0))
    # print(img_get) #三原色通道
    #
    # img_L=img.convert('L')
    # print(img_L)
    # img_get_L=img_L.getpixel((0,0))    #换算 得到灰度值
    # print(img_get_L)

    # 定义画笔的大小
    Pen_size = 3
    # 色差扩散器
    Color_Diff = 6
    for i in range(Pen_size + 1, width - Pen_size - 1):
        for j in range(Pen_size + 1, height - Pen_size - 1):
            # 原始的颜色
            originalColor = 255
            lcolor = sum([img.getpixel((i - r, j)) for r in range(Pen_size)]) // Pen_size
            rcolor = sum([img.getpixel((i + r, j)) for r in range(Pen_size)]) // Pen_size

            # 通道----颜料
            if abs(lcolor - rcolor) > Color_Diff:
                originalColor -= (255 - img.getpixel((i, j))) // 4
                new.putpixel((i, j), originalColor)

            ucolor = sum([img.getpixel((i, j - r)) for r in range(Pen_size)]) // Pen_size
            dcolor = sum([img.getpixel((i, j + r)) for r in range(Pen_size)]) // Pen_size

            # 通道----颜料
            if abs(ucolor - dcolor) > Color_Diff:
                originalColor -= (255 - img.getpixel((i, j))) // 4
                new.putpixel((i, j), originalColor)

            acolor = sum([img.getpixel((i - r, j - r)) for r in range(Pen_size)]) // Pen_size
            bcolor = sum([img.getpixel((i + r, j + r)) for r in range(Pen_size)]) // Pen_size

            # 通道----颜料
            if abs(acolor - bcolor) > Color_Diff:
                originalColor -= (255 - img.getpixel((i, j))) // 4
                new.putpixel((i, j), originalColor)

            qcolor = sum([img.getpixel((i + r, j - r)) for r in range(Pen_size)]) // Pen_size
            wcolor = sum([img.getpixel((i - r, j + r)) for r in range(Pen_size)]) // Pen_size

            # 通道----颜料
            if abs(qcolor - wcolor) > Color_Diff:
                originalColor -= (255 - img.getpixel((i, j))) // 4
                new.putpixel((i, j), originalColor)

    new.save(new_path)


gen_img('test.jpg', 'test1.jpg')

test_b_path = "C:\\Users\\aaa10\\Desktop\\电水壶\\testB"
test_a_path = "C:\\Users\\aaa10\\Desktop\\电水壶\\testA"
train_b_path = "C:\\Users\\aaa10\\Desktop\\电水壶\\trainB"
train_a_path = "C:\\Users\\aaa10\\Desktop\\电水壶\\trainA"

# if __name__ == '__main__':
#
#     pool = multiprocessing.Pool(processes = 10)
#     for i in os.listdir(test_b_path):
#         pool.apply_async(gen_img, (os.path.join(test_b_path, i), os.path.join(test_a_path, i),))  # 维持执行的进程总数为processes，当一个进程执行完毕后会添加新的进程进去
#
#     pool.close()
#     pool.join()  # 调用join之前，先调用close函数，否则会出错。执行完close后不会有新的进程加入到pool,join函数等待所有子进程结束
#
#     pool = multiprocessing.Pool(processes = 10)
#     for i in os.listdir(train_b_path):
#         pool.apply_async(gen_img, (os.path.join(train_b_path, i), os.path.join(train_a_path, i),))
#
#     pool.close()
#     pool.join()  # 调用join之前，先调用close函数，否则会出错。执行完close后不会有新的进程加入到pool,join函数等待所有子进程结束
