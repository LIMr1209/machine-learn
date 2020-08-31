from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
import os
from torchvision import datasets, transforms
import torch
from torch.nn import functional as F
from torch import nn
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import torchvision.models as models
from argparse import ArgumentParser
from pytorch_lightning.callbacks import EarlyStopping
from utils.utils import accuracy
from utils.image_loader import image_loader
 

from data.dataset import DatasetFromFilename
from models import efficientNet
from utils.get_classes import get_classes

seed_everything(42) # 为了确保每次运行都具有完全的可重复性，您需要为伪随机生成器设置种子，并在中设置deterministic`标志Trainer。


# save_last = True  每次 epoch 保存模型
checkpoint_callback = ModelCheckpoint(
    filepath=os.path.join(os.getcwd(),'checkpoint/{epoch}-{val_loss:.2f}'),
    save_top_k=True,  
    verbose=True,
    monitor='val_loss',  # 监控指标  为 最小
    mode='min',
    prefix='',
    save_weights_only=True # 保存权重
)
# 训练后 checkpoint_callback.best_model_path 获取最佳检查点

logger = TensorBoardLogger(
    save_dir=os.path.join(os.getcwd(),'runs'),
    name='lightning_logs'
)

# default used by the Trainer
early_stop_callback = EarlyStopping(
    monitor='val_loss',
    patience=3,
    strict=False,
    verbose=False,
    mode='min'
)

# 转移学习
class ImagenetTransferLearning(LightningModule):
    def __init__(self,hparams):
        super(ImagenetTransferLearning, self ).__init__() 
        self.hparams = hparams
        # init a pretrained resnet
        self.feature_extractor = efficientNet(pretrained=True, override_params={'num_classes': hparams.num_classes})
        # self.feature_extractor.eval()

        # use the pretrained model to classify cifar-10 (10 image classes)
        # self.classifier = nn.Linear(2048, hparams.num_classes)

    def forward(self, x):
        representations = self.feature_extractor(x)
        # x = self.classifier(representations)
        return representations
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--batch_size', type=int, default=12)
        parser.add_argument('--learning_rate', type=float, default=0.001)
        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument('--data_root', type=str, default='/home/thn/test_image')
        parser.add_argument('--max_epoch', type=int, default=25)
        parser.add_argument('--lr_gamma', type=float, default=0.5) # 学习效率下降 lr*lr_decay
        parser.add_argument('--lr_policy', type=str, default='multi') #   学习效率调度器  plateau,step,multi
        parser.add_argument('--lr_epoch', type=list, default=[3, 5, 7]) # 训练epoch达到milestones值时,初始学习率乘以gamma得到新的学习率
        parser.add_argument('--weight_decay', type=float, default=0e-5) # 优化器权值衰减率
        data = parser.parse_args()
        data.cate_classes = get_classes(data.data_root)['class2num'] # 图像分类标签列表
        data.num_classes = len(data.cate_classes)
        return data
    
    def prepare_data(self):
        self.train_dataset = DatasetFromFilename(self.hparams.data_root, flag='train')  # 训练集
        self.val_dataset = DatasetFromFilename(self.hparams.data_root, flag='test')  # 验证集
        self.test_dataset = DatasetFromFilename(self.hparams.data_root, flag='valid')  # 测试集
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers)
    
    def test_step(self, batch, batch_idx):
        x, y, a, b = batch
        logits = self(x)
        criterion = torch.nn.CrossEntropyLoss()  # 损失函数
        loss = criterion(logits, y)
        top1, top5 = accuracy(logits, y, topk=(1, 5))  # top1 和 top5 的准确率
        return {'val_loss': loss,'top1':top1, 'top5':top5}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_top1 = torch.stack([x['top1'] for x in outputs]).mean()
        avg_top5 = torch.stack([x['top5'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs,'top1':avg_top1,'top5':avg_top5}
    
    def validation_step(self, batch, batch_idx):
        x, y, a, b = batch
        logits = self(x)
        criterion = torch.nn.CrossEntropyLoss()  # 损失函数
        loss = criterion(logits, y)
        top1, top5 = accuracy(logits, y, topk=(1, 5))  # top1 和 top5 的准确率
        return {'val_loss': loss,'top1':top1, 'top5':top5}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_top1 = torch.stack([x['top1'] for x in outputs]).mean()
        avg_top5 = torch.stack([x['top5'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs,'top1':avg_top1,'top5':avg_top5}

    def configure_optimizers(self):
        return torch.optim.Adam(self.feature_extractor._fc.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)

    def training_step(self, batch, batch_idx):
        x, y, a, b = batch
        logits = self(x)
        criterion = torch.nn.CrossEntropyLoss()  # 损失函数
        loss = criterion(logits, y)
        # add logging
        tensorboard_logs = {'loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}
    

def recognition(url):
    with torch.no_grad():  # 用来标志计算要被计算图隔离出去,取消梯度
        image = image_loader(url)
        parser = ArgumentParser()
        hparams = ImagenetTransferLearning.add_model_specific_args(parser) # 超参数
        model = ImagenetTransferLearning.load_from_checkpoint(checkpoint_path="checkpoint/epoch=26-val_loss=0.15.ckpt")
        model.freeze() #  冻结
        image = image.view(1, 3, 224, 224)  # 转换image
        outputs = model(image)
        result = {}
        for i in range(hparams.num_classes):  # 计算各分类比重
            result[hparams.cate_classes[i]] = t.nn.functional.softmax(outputs, dim=1)[:, i].detach().tolist()[0]
            result = sorted(result.items(), key=lambda x: x[1], reverse=True)
        print(result)
if __name__ == '__main__':
    parser = ArgumentParser()
    hparams = ImagenetTransferLearning.add_model_specific_args(parser) # 超参数
    model = ImagenetTransferLearning(hparams)
    #resume_from_checkpoint=None
    # trainer = Trainer(gpus=1, benchmark=True, checkpoint_callback=checkpoint_callback, max_epochs=25, reload_dataloaders_every_epoch=True, deterministic=True,logger=logger, val_check_interval=1.0)
    trainer = Trainer(gpus=1, benchmark=True, checkpoint_callback=checkpoint_callback, max_epochs=27, reload_dataloaders_every_epoch=True, deterministic=True,resume_from_checkpoint='checkpoint/epoch=24-val_loss=0.15.ckpt',logger=logger,val_check_interval=1.0)
    trainer.fit(model)
    # new_model = ImagenetTransferLearning.load_from_checkpoint(checkpoint_path="checkpoint/epoch=26-val_loss=0.15.ckpt")
#     trainer = Trainer(gpus=1)
#     trainer.test(new_model)