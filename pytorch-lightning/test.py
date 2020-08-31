from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
import os
from torchvision import datasets, transforms
import torch
from torch.nn import functional as F
from torch import nn
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import torchvision.models as models
from argparse import ArgumentParser



# Trainer(logger=logger)
# default logger used by trainer
# logger = TensorBoardLogger(
#     save_dir=os.getcwd(),
#     version=self.slurm_job_id,
#     name='lightning_logs'
# )

# Trainer(checkpoint_callback=checkpoint_callback)
# default used by the Trainer
checkpoint_callback = ModelCheckpoint(
    filepath=os.getcwd(),
    save_best_only=True,
    verbose=True,
    monitor='val_loss',
    mode='min',
    prefix=''
)

# # Trainer(early_stop_callback=early_stop_callback)
# # default used by the Trainer
# early_stop_callback = EarlyStopping(
#     monitor='val_loss',
#     patience=3,
#     strict=False,
#     verbose=False,
#     mode='min'
# )

class MyPrintingCallback(Callback):

    def on_init_start(self, trainer):
        print('Starting to init trainer!')

    def on_init_end(self, trainer):
        print('Trainer is init now')

    def on_train_end(self, trainer, pl_module):
        print('do something when training ends')


        
        
class LitMNIST(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.layer_1 = torch.nn.Linear(28 * 28, 128)
        self.layer_2 = torch.nn.Linear(128, 256)
        self.layer_3 = torch.nn.Linear(256, 10)

    def forward(self, x):
        batch_size, channels, width, height = x.size()
        x = x.view(batch_size, -1)
        x = self.layer_1(x)
        x = torch.relu(x)
        x = self.layer_2(x)
        x = torch.relu(x)
        x = self.layer_3(x)
        x = torch.log_softmax(x, dim=1)
        return x
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--layer_1_dim', type=int, default=128)
        parser.add_argument('--layer_2_dim', type=int, default=256)
        parser.add_argument('--batch_size', type=int, default=64)
        parser.add_argument('--learning_rate', type=float, default=0.002)
        return parser
    
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

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=64)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=64)
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        return {'val_loss': loss}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)

        # add logging
        logs = {'loss': loss}
        return {'loss': loss, 'log': logs}


if __name__ == '__main__':
    parser = ArgumentParser()
    parser = LitMNIST.add_model_specific_args(parser) # 超参数
    hparams = parser.parse_args()
#     model = LitMNIST(hparams,callbacks=[MyPrintingCallback()])
    model = LitMNIST(hparams) 
    # benchmark true 如果输入大小不变，则此标志可能会提高系统速度
    # max_steps 最大训练  # max_epochs
    # fast_dev_run=True 该标志通过运行1个训练批次和1个验证批次来运行“单元测试”
    # num_sanity_val_steps  默认5  在开始训练例程之前，进行健全性检查运行n批val。这样可以捕获验证中的所有错误
    # reload_dataloaders_every_epoch 默认 false 设置为True可在每个时期重新加载数据加载器。
    # resume_from_checkpoint  要从特定检查点恢复训练  fit
    # deterministic=True  随机生成器设置种子
    # auto_scale_batch_size='batch_size' 自动尝试找到适合内存的最大批处理大小,翻盖设置的超参数batch_size
    # auto_lr_find='learning_rate' 自动尝试找到适合的lr,翻盖设置的超参数learning_rate
    trainer = Trainer(gpus=1,checkpoint_callback=checkpoint_callback)
    trainer.fit(model)
    # trainer.save_checkpoint("example.ckpt")
    # new_model = MyModel.load_from_checkpoint(checkpoint_path="example.ckpt")
    # run test set
    # model.freeze() # 冻结
    # trainer.test()
    # model = LitMNIST.load_from_checkpoint(PATH) # 加载模型
    # trainer = Trainer(num_tpu_cores=8)
    # trainer.test(model)
