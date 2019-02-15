import warnings
import torch as t

from utils.get_classes import get_classes


class DefaultConfig(object):
    env = 'opalus_recognltion'  # visdom 环境
    vis_port = 8097  # visdom 端口
    image_size = 224  # 图片尺寸
    model = 'AlexNet1'  # 使用的模型，名字必须与models/__init__.py中的名字一致

    data_root = "/home/tian/Desktop/spiders/design/design/spiders/image"  # 数据集存放路径
    load_model_path = None  # 加载训练的模型的路径，为None代表不加载
    # load_model_path = './checkpoint/AlexNet1_0214_10-06-50.pth.tar'

    batch_size = 16  # 每批训练数据的个数,显存不足,适当减少
    use_gpu = True  # user GPU or not
    num_workers = 4  # how many workers for loading data
    print_freq = 2  # print info every N batch
    vis = True  # 是否使用visdom可视化

    cate_classes = get_classes(data_root)['class2num']  # 分类列表
    num_classes = len(cate_classes)  # 分类个数
    # pretrained = False  # 不加载预训练
    pretrained = True  # 加载预训练模型

    max_epoch = 20  # 学习次数
    lr = 0.001  # initial learning rate
    lr_decay = 0.5  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 0e-5  # 损失函数
    # url = 'https://ss3.bdstatic.com/70cFv8Sh_Q1YnxGkpoWK1HF6hhy/it/u=614134999,3540271868&fm=27&gp=0.jpg'  # 识别图片地址
    # url = 'https://ss1.bdstatic.com/70cFuXSh_Q1YnxGkpoWK1HF6hhy/it/u=688429408,3192272581&fm=27&gp=0.jpg'
    url = 'https://ss1.bdstatic.com/70cFvXSh_Q1YnxGkpoWK1HF6hhy/it/u=1515206672,3808938099&fm=27&gp=0.jpg'

    # url = 'https://ss0.bdstatic.com/70cFuHSh_Q1YnxGkpoWK1HF6hhy/it/u=3211343338,3677737612&fm=27&gp=0.jpg'
    # url = 'https://ss0.bdstatic.com/70cFuHSh_Q1YnxGkpoWK1HF6hhy/it/u=1173573129,2720567755&fm=27&gp=0.jpg'

    def _parse(self, kwargs):
        """
        根据字典kwargs 更新 config参数
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)

        opt.device = t.device('cuda') if opt.use_gpu else t.device('cpu')

        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                print(k, getattr(self, k))


opt = DefaultConfig()
