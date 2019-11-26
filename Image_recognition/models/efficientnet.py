from utils.utils import init_net
from .basic_module import BasicModule
from config import opt
from efficientnet_pytorch import EfficientNet as ef
import torch


def efficientNet(pretrained, override_params=None):
    model = ef.from_name('efficientnet-b5', override_params=override_params)
    if pretrained:
        pretrained_state_dict = torch.load('Authority/efficientnet-b5-586e6cc6.pth')
        now_state_dict = model.state_dict()  # 返回model模块的字典
        pretrained_state_dict.pop('_fc.weight')  # 排除全连接层的参数(全连接层返回分类个数)
        pretrained_state_dict.pop('_fc.bias')
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(
            now_state_dict)
    else:
        init_net(model)
    return model


class EfficientNet(BasicModule):
    def __init__(self):
        super(EfficientNet, self).__init__()
        self.model_name = 'EfficientNet'
        self.model = efficientNet(pretrained=opt.pretrained, override_params={'num_classes': opt.num_classes})

    def forward(self, x):
        x = self.model(x)

        return x

    def get_optimizer(self, lr, weight_decay):
        if not opt.pretrained:
            return super(EfficientNet, self).get_optimizer(lr, weight_decay)
        else:
            return torch.optim.Adam(self.model._fc.parameters(), lr=lr, weight_decay=weight_decay)


if __name__ == '__main__':
    a = EfficientNet().to(torch.device('cuda'))
    input = torch.autograd.Variable(torch.randn(6, 3, 224, 224)).to(torch.device('cuda'))
    output = a(input)
    print(output.size())
