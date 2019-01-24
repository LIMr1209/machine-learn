from .basic_module import BasicModule
import torch.nn as nn
from config import opt
from torchvision.models import ResNet
import torch
from torch.utils.checkpoint import checkpoint


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
        pretrained_state_dict.pop('fc.weight')
        pretrained_state_dict.pop('fc.bias')
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(
            now_state_dict)  # 最后通过调用model的load_state_dict方法用预训练的模型参数来初始化你构建的网络结构，这个方法就是PyTorch中通用的用一个模型的参数初始化另一个模型的层的操作。load_state_dict方法还有一个重要的参数是strict，该参数默认是True，表示预训练模型的层和你的网络结构层严格对应相等（比如层名和维度）
        return model
    return ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)


class ResNet152(BasicModule):
    def __init__(self):
        super(ResNet152, self).__init__()
        self.model_name = 'ResNet152'
        self.model = resnet152(pretrained=opt.pretrained, num_classes=opt.num_classes)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.model.fc(x)

        return x

    def get_optimizer(self, lr, weight_decay):
        if not opt.pretrained:
            return super(ResNet152, self).get_optimizer(lr, weight_decay)
        else:
            return torch.optim.Adam(self.model.fc.parameters(), lr=lr, weight_decay=weight_decay)


if __name__ == '__main__':
    a = ResNet152().to(torch.device('cuda'))
    input = torch.autograd.Variable(torch.randn(16, 3, 224, 224)).to(torch.device('cuda'))
    output = a(input)
    print(output.size())
