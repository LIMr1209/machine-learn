import os
import sys
from PIL import Image
import multiprocessing

def img_merge(a,b,c):
    im1 = Image.open(a)
    im2 = Image.open(b)
    im3 = Image.new('RGB', (im1.width*2, im1.height))
    im3.paste(im1,(0,0,im1.width,im1.height))
    im3.paste(im2,(im2.width,0,im2.width*2,im2.height))
    im3.save(c)

if __name__ == '__main__':
    base_path = "C:\\Users\\aaa10\\Desktop\\"
    test_b_path = os.path.join(base_path,sys.argv[1],"testB")
    test_a_path = os.path.join(base_path,sys.argv[1],"testA")
    train_b_path = os.path.join(base_path,sys.argv[1],"trainB")
    train_a_path = os.path.join(base_path,sys.argv[1],"trainA")
    pix2pix_path_train = os.path.join(sys.argv[1]+'_pix2pix',"train")
    pix2pix_path_val = os.path.join(sys.argv[1]+'_pix2pix',"val")
    if not os.path.exists(pix2pix_path_train):
        os.makedirs(pix2pix_path_train)
    if not os.path.exists(pix2pix_path_val):
        os.makedirs(pix2pix_path_val)
    pool = multiprocessing.Pool(processes = 10)
    for i, j in enumerate(os.listdir(test_b_path)):
        pool.apply_async(img_merge, (os.path.join(test_a_path, j), os.path.join(test_b_path, j),os.path.join(pix2pix_path_val, str(i)+'.jpg'),))  # 维持执行的进程总数为processes，当一个进程执行完毕后会添加新的进程进去
    pool.close()
    pool.join()  # 调用join之前，先调用close函数，否则会出错。执行完close后不会有新的进程加入到pool,join函数等待所有子进程结束
    pool = multiprocessing.Pool(processes = 10)
    for i, j in enumerate(os.listdir(train_b_path)):
        pool.apply_async(img_merge, (os.path.join(train_a_path, j), os.path.join(train_b_path, j),os.path.join(pix2pix_path_train, str(i)+'.jpg'),))

    pool.close()
    pool.join()  # 调用join之前，先调用close函数，否则会出错。执行完close后不会有新的进程加入到pool,join函数等待所有子进程结束

