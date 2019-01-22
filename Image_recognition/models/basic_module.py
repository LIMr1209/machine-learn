# coding:utf8
import torch as t
from utils.utils import save_checkpoint


class BasicModule(t.nn.Module):
    """
    封装了nn.Module,主要是提供了save和load两个方法
    """

    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))  # 默认名字

    def load(self, path):
        """
        可加载指定路径的模型
        """

        # 加载模型
        checkpoint = t.load(path)
        return checkpoint

    def save(self, state):
        """
        保存模型，默认使用“模型名字+时间”作为文件名
        """
        save_checkpoint(state)

    def get_optimizer(self, lr, weight_decay):  # 权重衰减  防止过拟合
        # return t.optim.SGD(self.parameters(), lr=lr, momentum=0.9) # 随机梯度下降法
        # return t.optim.Adagrad(self.parameters(),lr=lr,lr_decay,weight_decay=weight_decay ) # 自适应梯度法 自动变更学习速率,
        # return t.optim.Adadelta(self.parameters(),lr=lr, weight_decay=weight_decay) # 完全自适应全局学习率，加速效果好
        # return t.optim.RMSprop(self.parameters(), lr=lr, alpha=0.9, weight_decay=weight_decay)
        # 其实它就是Adadelta，这里的RMS就是Adadelta中定义的RMS，也有人说它是一个特例，ρ=0.5的Adadelta，且分子α，即仍然依赖于全局学习率
        return t.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)  # adam是世界上最好的优化算法，不知道用啥时，用它就对了


class Flat(t.nn.Module):
    """
    把输入reshape成（batch_size,dim_length）
    """

    def __init__(self):
        super(Flat, self).__init__()
        # self.size = size

    def forward(self, x):
        return x.view(x.size(0), -1)
