from .basic_module import BasicModule
from config import opt
from torchvision.models import AlexNet
import torch
import torch.nn as nn


def alexnet(pretrained=False, **kwargs):  # 224*224
    if pretrained:
        model = AlexNet(**kwargs)
        pretrained_state_dict = torch.load(
            './Authority/alexnet-owt-4df8aa71.pth')
        now_state_dict = model.state_dict()  # 返回model模块的字典
        pretrained_state_dict.pop('classifier.6.weight')
        pretrained_state_dict.pop('classifier.6.bias')
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(
            now_state_dict)
        return model
    return AlexNet(**kwargs)


class AlexNet1(BasicModule):
    def __init__(self):
        super(AlexNet1, self).__init__()
        self.model_name = 'AlexNet1'
        self.model = alexnet(pretrained=opt.pretrained, num_classes=opt.num_classes)

    def forward(self, x):
        return self.model(x)

    def get_optimizer(self, lr, weight_decay):
        if not opt.pretrained:
            return super(AlexNet1, self).get_optimizer(lr, weight_decay)
        else:
            return torch.optim.Adam(self.model.classifier[6].parameters(), lr=lr, weight_decay=weight_decay)


if __name__ == '__main__':
    a = AlexNet1()
    for i in a.model.classifier[6]:
        print(i)
