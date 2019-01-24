from .basic_module import BasicModule
from config import opt
from torchvision.models import Inception3
import torch


def inception_v3(pretrained=False, **kwargs):  # 299*299
    if pretrained:
        if 'transform_input' not in kwargs:
            kwargs['transform_input'] = True
        model = Inception3(**kwargs)
        pretrained_state_dict = torch.load(
            './Authority/inception_v3_google-1a9a5a14.pth')  # load_url函数根据model_urls字典下载或导入相应的预训练模型
        now_state_dict = model.state_dict()  # 返回model模块的字典
        pretrained_state_dict.pop('AuxLogits.fc.weight')
        pretrained_state_dict.pop('AuxLogits.fc.bias')
        pretrained_state_dict.pop('fc.weight')
        pretrained_state_dict.pop('fc.bias')
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(
            now_state_dict)  # 最后通过调用model的load_state_dict方法用预训练的模型参数来初始化你构建的网络结构，这个方法就是PyTorch中通用的用一个模型的参数初始化另一个模型的层的操作。load_state_dict方法还有一个重要的参数是strict，该参数默认是True，表示预训练模型的层和你的网络结构层严格对应相等（比如层名和维度）
        return model
    return Inception3(**kwargs)


class InceptionV3(BasicModule):
    def __init__(self):
        super(InceptionV3, self).__init__()
        self.model_name = 'InceptionV3'
        self.model = inception_v3(pretrained=opt.pretrained, num_classes=opt.num_classes)

    def forward(self, x):
        return self.model(x)

    def get_optimizer(self, lr, weight_decay):
        if not opt.pretrained:
            return super(InceptionV3, self).get_optimizer(lr, weight_decay)
        else:
            return torch.optim.Adam([
                {'params': self.model.AuxLogits.fc.parameters()},
                {'params': self.model.fc.parameters()}
            ], lr=lr, weight_decay=weight_decay)


if __name__ == '__main__':
    a = InceptionV3()
    print(a)
