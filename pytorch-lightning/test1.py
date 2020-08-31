import torchvision.models as models
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
import os
from torchvision import datasets, transforms
import torch
from torch.nn import functional as F


class ImagenetTransferLearning(LightningModule):
    def __init__(self):
        super(ImagenetTransferLearning, self ).__init__() 
        # init a pretrained resnet
        num_target_classes = 1000
        self.feature_extractor = models.resnet50(pretrained=True,num_classes=num_target_classes)
        self.feature_extractor.eval()

        # use the pretrained model to classify cifar-10 (10 image classes)
        self.classifier = nn.Linear(2048, num_target_classes)
    def prepare_data(self):
        # transform
        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

        # download
        mnist_train = MNIST(os.getcwd(), train=True, download=True, transform=transform)
        mnist_test = MNIST(os.getcwd(), train=False, download=True, transform=transform)

        # train/val split
        mnist_train, mnist_val = random_split(mnist_train, [55000, 5000])

        # assign to use in dataloaders
        self.train_dataset = mnist_train
        self.val_dataset = mnist_val
        self.test_dataset = mnist_test
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=64)

    def forward(self, x):
        representations = self.feature_extractor(x)
        x = self.classifier(representations)
        return x
    
    
    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)
    
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        criterion = t.nn.CrossEntropyLoss().to(opt.device)  # 损失函数
        loss = criterion(logits, y)
        # add logging
        logs = {'loss': loss}
        return {'loss': loss, 'log': logs}
 
if __name__ == '__main__':
    model = ImagenetTransferLearning()
    trainer = Trainer(gpus=1)
    trainer.fit(model)