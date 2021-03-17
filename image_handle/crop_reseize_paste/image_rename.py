import os
import sys
import multiprocessing

def img_rename(a,b,mode):
    if mode == 'test':
        os.rename(os.path.join(test_a_path,a),os.path.join(test_a_path,b))
        os.rename(os.path.join(test_b_path,a),os.path.join(test_b_path,b))
    else:
        os.rename(os.path.join(train_a_path,a),os.path.join(train_a_path,b))
        os.rename(os.path.join(train_b_path,a),os.path.join(train_b_path,b))

if __name__ == '__main__':
    base_path = "C:\\Users\\aaa10\\Desktop\\"
    test_b_path = os.path.join(base_path, sys.argv[1],"testB")
    test_a_path = os.path.join(base_path, sys.argv[1],"testA")
    train_b_path = os.path.join(base_path, sys.argv[1],"trainB")
    train_a_path = os.path.join(base_path, sys.argv[1],"trainA")
    pool = multiprocessing.Pool(processes = 10)
    for i, j in enumerate(os.listdir(test_b_path)):
        pool.apply_async(img_rename, (j,str(i)+'.jpg','test', ))  # 维持执行的进程总数为processes，当一个进程执行完毕后会添加新的进程进去
    pool.close()
    pool.join()  # 调用join之前，先调用close函数，否则会出错。执行完close后不会有新的进程加入到pool,join函数等待所有子进程结束
    pool = multiprocessing.Pool(processes = 10)
    for i, j in enumerate(os.listdir(train_b_path)):
        pool.apply_async(img_rename, (j,str(i)+'.jpg','train', ))

    pool.close()
    pool.join()  # 调用join之前，先调用close函数，否则会出错。执行完close后不会有新的进程加入到pool,join函数等待所有子进程结束

