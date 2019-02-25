import warnings
import torch as t

from utils.get_classes import get_classes


class DefaultConfig(object):
    env = 'opalus_recognltion'  # visdom 环境
    vis_port = 8097  # visdom 端口
    image_size = 224  # 图片尺寸
    model = 'ResNet152'  # 使用的模型，名字必须与models/__init__.py中的名字一致

    data_root = "/home/tian/Desktop/image"  # 数据集存放路径
    load_model_path = None  # 加载训练的模型的路径，为None代表不加载
    # load_model_path = './checkpoint/ResNet152_0221_17-28-30.pth.tar'

    batch_size = 16  # 每批训练数据的个数,显存不足,适当减少
    use_gpu = True  # user GPU or not
    num_workers = 4  # how many workers for loading data
    print_freq = 2  # print info every N batch
    vis = False  # 是否使用visdom可视化

    cate_classes = get_classes(data_root)['class2num']  # 分类列表
    num_classes = len(cate_classes)  # 分类个数
    # pretrained = False  # 不加载预训练
    pretrained = True  # 加载预训练模型

    max_epoch = 15  # 学习次数
    lr = 0.001  # initial learning rate
    lr_decay = 0.5  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 0e-5  # 损失函数
    # url = 'https://ss0.bdstatic.com/70cFuHSh_Q1YnxGkpoWK1HF6hhy/it/u=3566530008,2476414322&fm=26&gp=0.jpg'
    url = 'https://ss2.bdstatic.com/70cFvnSh_Q1YnxGkpoWK1HF6hhy/it/u=304436698,3711210886&fm=26&gp=0.jpg'



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
