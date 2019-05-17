## 环境准备

- 本程序需要安装[PyTorch](https://pytorch.org/) 本人安装总结[地址](https://blog.csdn.net/qq_41654985/article/details/86599016)
- 还需要通过`pip install -r requirements.txt` 安装其它依赖

## 可视化

- 如果想要使用tensorboard可视化,请先运行`tensorboard --logdir=runs`启动tensorboard服务

## 训练
然后使用如下命令启动训练：

```
python main.py train --env='test' --use-gpu --print-freq=4
```
可选参数
- image_size = 224  # 图片尺寸
- model = 'ResNet152'  # 使用的模型，名字必须与models/__init__.py中的名字一致
- data_root = "/home/tian/Desktop/image"  # 数据集存放路径
- load_model_path = None  # 加载训练的模型的路径，为None代表不加载
- batch_size = 16  # 每批训练数据的个数,显存不足,适当减少
- use_gpu = True  # 是否使用GPU
- num_workers = 4  # 用于数据预处理的多处理工作器的数量
- print_freq = 2  # 数据可视化指数
- vis = False  # 是否使用tensorboard可视化
- pretrained = True  # 加载预训练模型
- max_epoch = 25  # 学习次数
- lr = 0.001  # 学习效率
- lr_decay = 0.5  # 误差增加时,学习效率下降
- weight_decay = 0e-5  # 损失函数
- error_img = 'error_img.csv' # 测试图片错误统计
- url = '图片地址'

修剪量化参数,敏感性分析
- pruning = False  # 是否修剪
- compress = None  # 压缩计划表
- quantize_eval = False
- qe_calibration = None
- qe_mode = LinearQuantMode.SYMMETRIC
- qe_bits_acts = 8
- qe_bits_wts = 8
- qe_bits_accum = 32
- qe_clip_acts = False
- qe_no_clip_layers = []
- qe_per_channel = False
- qe_stats_file = None
- qe_config_file = None
- output_dir = 'logs'  # 日志输出
- name = 'opalus_recognltion'  # 实验名
- sensitivity = 'element'  # ['element', 'filter', 'channel']  # 神经网络敏感性分析
- sensitivity_range = [0.4, 0.9, 0.1]  # 尝试修剪比例

## 测试

```
python main.py test --load-path='checkpoints/ResNet152.pth.tar'
```

## 识别

```
python main.py recognition --url='图片地址' --load-path='checkpoints/ResNet152.pth.tar'
```
