import torch.nn as nn
from torchvision.models import ResNet
import torch
import re
pretrained = True

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def resnet152(pretrained=False, **kwargs):
    if pretrained:
        model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
        pretrained_state_dict = torch.load(
            './Authority/resnet152-b121ed2d.pth')  # load_url函数根据model_urls字典下载或导入相应的预训练模型
        now_state_dict = model.state_dict()  # 返回model模块的字典
        pretrained_state_dict.pop('fc.weight')  # 排除全连接层的参数(全连接层返回分类个数)
        pretrained_state_dict.pop('fc.bias')
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(
            now_state_dict)
        # 最后通过调用model的load_state_dict方法用预训练的模型参数来初始化你构建的网络结构，
        # 这个方法就是PyTorch中通用的用一个模型的参数初始化另一个模型的层的操作。load_state_dict方法还有一个重要的参数是strict，
        # 该参数默认是True，表示预训练模型的层和你的网络结构层严格对应相等（比如层名和维度）
        return model
    return ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)


class ResNet152(nn.Module):
    def __init__(self):
        super(ResNet152, self).__init__()
        self.model_name = 'ResNet152'
        self.model = resnet152(pretrained=pretrained, num_classes=6)

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    model = ResNet152().to(torch.device('cuda'))
    input = torch.autograd.Variable(torch.randn(16, 3, 224, 224)).to(torch.device('cuda'))
    params = model.state_dict()
    rex = re.compile(r'.*(conv|downsample\.|fc)\d?\.weight')
    for i in params.keys():
        if rex.search(i):
            print(i)

