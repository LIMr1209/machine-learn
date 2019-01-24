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
        state_dict = torch.load('./checkpoint/inception_v3_google-1a9a5a14.pth')
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict)
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


if __name__ == '__main__':
    a = DenseNet161()
    print(a)
