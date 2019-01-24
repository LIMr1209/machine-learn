from .basic_module import BasicModule
from config import opt
from torchvision.models import DenseNet
import re
import torch


def densenet201(pretrained=False, **kwargs):
    if pretrained:
        model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 48, 32),
                         **kwargs)
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = torch.load('./checkpoint/inception_v3_google-1a9a5a14.pth')
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict)
        return model
    return DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 48, 32),
                    **kwargs)


class DenseNet201(BasicModule):
    def __init__(self):
        super(DenseNet201, self).__init__()
        self.model_name = 'DenseNet201'
        self.model = densenet201(pretrained=opt.pretrained, num_classes=opt.num_classes)

    def forward(self, x):
        return self.model(x)

    def get_optimizer(self, lr, weight_decay):
        if not opt.pretrained:
            return super(DenseNet201, self).get_optimizer(lr, weight_decay)
        else:
            return torch.optim.Adam(self.model.fc.parameters(), lr=lr, weight_decay=weight_decay)


if __name__ == '__main__':
    a = DenseNet201()
    print(a)
