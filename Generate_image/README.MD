## 环境准备

- 本程序需要安装[PyTorch](https://pytorch.org/) 本人安装总结[地址](https://blog.csdn.net/qq_41654985/article/details/86599016)
- 还需要通过`pip install -r requirements.txt` 安装其它依赖

## 数据
标签文件夹下放入图片

## 用法
如果想要使用visdom可视化，请先运行`python -m visdom.server`启动visdom服务

基本用法：
```
Usage： python main.py FUNCTION --key=value,--key2=value2 ..
```

- 训练
```bash
python main.py train --gpu --vis=False
```

- 生成图片

```bash
python main.py generate --nogpu --vis=False \
            --netd-path=checkpoints/netd_200.pth \
            --netg-path=checkpoints/netg_200.pth \
            --gen-img=result.png \
            --gen-num=64
```
完整的选项及默认值
```python
    data_path = 'data/' # 数据集存放路径
    num_workers = 4 # 多进程加载数据所用的进程数
    image_size = 96 # 图片尺寸
    batch_size = 256
    max_epoch =  200
    lr1 = 2e-4 # 生成器的学习率
    lr2 = 2e-4 # 判别器的学习率
    beta1=0.5 # Adam优化器的beta1参数
    gpu=True # 是否使用GPU --nogpu或者--gpu=False不使用gpu
    nz=100 # 噪声维度
    ngf = 64 # 生成器map数
    ndf = 64 # 判别器map数
    
    save_path = 'imgs/' #训练时生成图片保存路径
    
    vis = True # 是否使用visdom可视化
    env = 'opalus_generate' # visdom的env
    plot_every = 20 # 每间隔20 batch，visdom画图一次

    d_every=1 # 每1个batch训练一次判别器
    g_every=5 # 每5个batch训练一次生成器
    decay_every=10 # 每10个epoch保存一次模型
    netd_path = 'checkpoints/netd_211.pth' #预训练模型
    netg_path = 'checkpoints/netg_211.pth'
    
    # 只测试不训练
    gen_img = 'result.png'
    # 从512张生成的图片中保存最好的64张
    gen_num = 64 
    gen_search_num = 512 
    gen_mean = 0 # 噪声的均值
    gen_std = 1 #噪声的方差
```
### 兼容性测试
train 
- [x] GPU  
- [] CPU  
- [] Python2
- [x] Python3

test: 

- [x] GPU
- [] CPU
- [] Python2
- [x] Python3
