from torch import nn
from .basic_module import BasicModule
from config import opt
from torchvision.models import alexnet


class AlexNet(BasicModule):
    def __init__(self, model_type=opt.pretrained, num_classes=opt.num_classes):
        super(AlexNet, self).__init__()
        self.model_name = 'AlexNet'
        self.model_type = model_type
        if model_type:
            model = alexnet(pretrained=True)
            self.features = model.features
            model.classifier[6] = nn.Linear(4096,num_classes)
            self.classifier = model.classifier
        else:
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(64, 192, kernel_size=5, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(192, 384, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
            )
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    a = AlexNet()
    print(a)
