import warnings
import torch as t


class Config(object):
    data_path = 'data/'  # 数据集存放路径
    num_workers = 4  # 多进程加载数据所用的进程数
    image_size = 96  # 图片尺寸
    batch_size = 32
    max_epoch = 4000
    lr1 = 0.001  # 2e-4  # 生成器的学习率 2.7*2-4
    lr2 = 0.001  # 2e-4  # 判别器的学习率
    beta1 = 0.5  # Adam优化器的beta1参数
    use_gpu = True  # 是否使用GPU
    nz = 100  # 噪声维度
    ngf = 64  # 生成器 map数
    ndf = 64  # 判别器 map数
    save_path = 'imgs/'  # 生成图片保存路径

    vis = True  # 是否使用visdom可视化
    env = 'opalus_generate'  # visdom的env
    plot_every = 5  # 每间隔20 batch，visdom画图一次

    d_every = 1  # 每1个batch训练一次判别器
    g_every = 5  # 每5个batch训练一次生成器
    save_every = 500  # 每10个epoch保存一次模型
    netd_path = None
    netg_path = None
    # netd_path = 'checkpoints/netd_199.pth' #预训练判别模型
    # netg_path = 'checkpoints/netg_199.pth' #预训练生成模型
    # 只测试不训练
    gen_img = 'result.png'
    # 从512张生成的图片中保存最好的64张
    gen_num = 64
    gen_search_num = 512
    gen_mean = 0  # 噪声的均值
    gen_std = 1  # 噪声的方差

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


opt = Config()
