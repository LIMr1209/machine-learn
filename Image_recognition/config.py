import warnings
import torch as t
from distiller.quantization.range_linear import LinearQuantMode
from utils.get_classes import get_classes


class DefaultConfig(object):
    image_size = 224  # 图片尺寸
    model = 'ResNet152'  # 使用的模型，名字必须与models/__init__.py中的名字一致

    data_root = "/image/image"  # 数据集存放路径
    load_model_path = None  # 加载训练的模型的路径，为None代表不加载
    # load_model_path = './checkpoint/ResNet152.pth.tar.bak'  # 加载训练的模型的路径，为None代表不加载
    batch_size = 16  # 每批训练数据的个数,显存不足,适当减少
    use_gpu = True  # 是否使用GPU
    num_workers = 4  # 用于数据预处理的多处理工作器的数量
    print_freq = 100  # 数据可视化指数
    vis = False  # 是否使用tensorboard可视化

    cate_classes = get_classes(data_root)['class2num']  # 图像分类标签列表
    num_classes = len(cate_classes)  # 图像分类个数
    # pretrained = False  # 不加载预训练
    pretrained = True  # 加载预训练模型
    # pruning = True   # 压缩计划表
    pruning = False  # 是否修剪
    # compress = 'resnet152.schedule_sensitivity.yaml'  # 压缩计划表
    compress = None  # 压缩计划表
    # 量化 参数
    quantize_eval = False
    qe_calibration = None
    qe_mode = LinearQuantMode.SYMMETRIC
    qe_bits_acts = 8
    qe_bits_wts = 8
    qe_bits_accum = 32
    qe_clip_acts = False
    qe_no_clip_layers = []
    qe_per_channel = False
    qe_stats_file = None
    qe_config_file = None
    output_dir = 'logs'  # 日志输出
    name = 'opalus_recognltion'  # 实验名
    sensitivity = 'element'  # ['element', 'filter', 'channel']  # 神经网络敏感性分析
    sensitivity_range = [0.4, 0.9, 0.1]  # 尝试修剪比例

    max_epoch = 25  # 学习次数
    lr = 0.001  # 学习效率
    lr_decay = 0.5  # 误差增加时,学习效率下降
    weight_decay = 0e-5  # 损失函数
    url = 'https://imgservice1.suning.cn/uimg1/b2c/image/nc5F5_pjiXv5sYaX2Hrx4w.jpg_800w_800h_4e'  # 识别图片
    error_img = 'error_img.csv'

    def _parse(self, kwargs):
        """
        根据字典kwargs 更新 config参数
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)

        opt.device = t.device('cuda') if opt.use_gpu else t.device('cpu')

        # print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                pass
                # print(k, getattr(self, k))


opt = DefaultConfig()
