from .basic_module import BasicModule
from config import opt
from torchvision.models import Inception3
import torch


def inception_v3(pretrained=False, **kwargs):
    if pretrained:
        if 'transform_input' not in kwargs:
            kwargs['transform_input'] = True
        model = Inception3(**kwargs)
        model.load_state_dict(torch.load('./checkpoint/inception_v3_google-1a9a5a14.pth'))

    return Inception3(**kwargs)


class InceptionV3(BasicModule):
    def __init__(self):
        super(InceptionV3, self).__init__()
        self.model_name = 'InceptionV3'
        self.model = inception_v3(pretrained=opt.pretrained, num_classes=opt.num_classes)

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    a = InceptionV3()
    print(a)
