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
