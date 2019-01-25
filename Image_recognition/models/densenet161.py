from .basic_module import BasicModule
from config import opt
from torchvision.models import DenseNet
import re
import torch


def densenet161(pretrained=False, **kwargs):
    if pretrained:
        model = DenseNet(num_init_features=96, growth_rate=48, block_config=(6, 12, 36, 24),
                         **kwargs)
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        pretrained_state_dict = torch.load(
            './Authority/densenet161-8d451a50.pth')  # load_url函数根据model_urls字典下载或导入相应的预训练模型
        for key in list(pretrained_state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                pretrained_state_dict[new_key] = pretrained_state_dict[key]
                del pretrained_state_dict[key]
        now_state_dict = model.state_dict()  # 返回model模块的字典
        pretrained_state_dict.pop('classifier.weight')
        pretrained_state_dict.pop('classifier.bias')
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(
            now_state_dict)
        # 最后通过调用model的load_state_dict方法用预训练的模型参数来初始化你构建的网络结构，
        # 这个方法就是PyTorch中通用的用一个模型的参数初始化另一个模型的层的操作。load_state_dict方法还有一个重要的参数是strict，
        # 该参数默认是True，表示预训练模型的层和你的网络结构层严格对应相等（比如层名和维度）
        return model
    return DenseNet(num_init_features=96, growth_rate=48, block_config=(6, 12, 36, 24),
                    **kwargs)


class DenseNet161(BasicModule):
    def __init__(self):
        super(DenseNet161, self).__init__()
        self.model_name = 'DenseNet161'
        self.model = densenet161(pretrained=opt.pretrained, num_classes=opt.num_classes)

    def forward(self, x):
        return self.model(x)

    def get_optimizer(self, lr, weight_decay):
        if not opt.pretrained:
            return super(DenseNet161, self).get_optimizer(lr, weight_decay)
        else:
            return torch.optim.Adam(self.model.fc.parameters(), lr=lr, weight_decay=weight_decay)


if __name__ == '__main__':
    a = DenseNet161()
    print(a)
