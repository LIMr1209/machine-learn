from .basic_module import BasicModule
from config import opt
from torchvision.models import SqueezeNet
import torch


def squeezenet1_1(pretrained=False, **kwargs):
    if pretrained:
        model = SqueezeNet(version=1.1, **kwargs)
        model.load_state_dict(torch.load('./checkpoint/inception_v3_google-1a9a5a14.pth'))
        return model
    return SqueezeNet(version=1.1, **kwargs)


class SqueezeNet1_1(BasicModule):
    def __init__(self):
        super(SqueezeNet1_1, self).__init__()
        self.model_name = 'SqueezeNet1_1'
        self.model = squeezenet1_1(pretrained=opt.pretrained, num_classes=opt.num_classes)

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    a = SqueezeNet1_1()
    print(a)
