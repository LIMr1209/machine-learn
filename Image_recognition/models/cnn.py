import torch.nn as nn



# CNN 卷积神经网络
# torchvision.models  SqueezeNet  轻量化卷积神经网络
# 要研究CNN类型DL网络模型在图像分类上的应用，就逃不开研究alexnet，这是CNN在图像分类上的经典模型。
# torchvision.models alexnet
# 经典 cnn VGGNet  torchvision.models vggnet
# DRN 深度残差网络
# torchvision.models  ResNet
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            # 灰度图 1 彩色图 3
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),  # 二维卷积层
            # weight(tensor) - 卷积的权重，shape是(out_channels, in_channels,kernel_size)`
            # bias(tensor) - 卷积的偏置系数，shape是（out_channel）
            nn.Dropout2d(0.5),  # 掉落50％的神经元
            nn.BatchNorm2d(64),  # 归一化  批标准化
            # 其实就是把越来越偏的分布强制拉回比较标准的分布，这样使得激活输入值落在非线性函数对输入比较敏感的区域，
            # 这样输入的小变化就会导致损失函数较大的变化，意思是这样让梯度变大，避免梯度消失问题产生，而且梯度变大意味着学习收敛速度快，能大大加快训练速度。
            nn.ReLU(),  # 激励函数  激活函数
            # 对输入运用修正线性单元函数${ReLU}(x)= max(0, x)$
            nn.MaxPool2d(kernel_size=3, stride=2))  # 向下采样  池化层
        # 对于输入信号的输入通道，提供2维最大池化（max pooling）操作
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.Dropout(0.5),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.fc = nn.Linear(2 * 12 * 192, num_classes)  # 全连接层  输出分类

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


# 前馈神经网络


# DNN 深度神经网络


# RNN 循环神经网络  分类


# RNN 循环神经网络  回归


# GAN 生成对抗网络


# VAE 自编码 （非监督）

# DCGAN 标签生成图片

# A3C 强化学习


# DRN 强化学习


if __name__ == '__main__':
    pass
    # cnn = ConvNet(input)
