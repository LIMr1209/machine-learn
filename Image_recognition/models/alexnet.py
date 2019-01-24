from .basic_module import BasicModule
from config import opt
from torchvision.models import AlexNet
import torch


def alexnet(pretrained=False, **kwargs): # 224*224
    if pretrained:
        model = AlexNet(**kwargs)
        model.load_state_dict(torch.load('./checkpoint/inception_v3_google-1a9a5a14.pth'))
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
            return torch.optim.Adam(self.model.fc.parameters(), lr=lr, weight_decay=weight_decay)


if __name__ == '__main__':
    a = AlexNet1()
    print(a)
