import math
import os
import operator
from datetime import datetime
import distiller
from distiller import apputils
from distiller.data_loggers import *
from config import opt
import torch as t
import models
from data.dataset import DatasetFromFilename
from torch.utils.data import DataLoader
from utils.image_loader import image_loader
from utils.utils import AverageMeter, accuracy, write_err_img
from utils.sensitivity import sensitivity_analysis, val
from utils.progress_bar import ProgressBar
from tqdm import tqdm
import numpy as np
import distiller.quantization as quantization
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

seed = 1000
t.cuda.manual_seed(seed)  # 随机数种子,当使用随机数时,关闭进程后再次生成和上次得一样


class Classifier:
    def __init__(self, **kwargs):
        opt._parse(kwargs)
        self.opt = opt
        self.model = getattr(models, self.opt.model)()
        self.criterion = t.nn.CrossEntropyLoss().to(self.opt.device)
        # 1. 铰链损失（Hinge Loss）：主要用于支持向量机（SVM） 中；
        # 2. 互熵损失 （Cross Entropy Loss，Softmax Loss ）：用于Logistic 回归与Softmax 分类中；
        # 3. 平方损失（Square Loss）：主要是最小二乘法（OLS）中；
        # 4. 指数损失（Exponential Loss） ：主要用于Adaboost 集成学习算法中；
        # 5. 其他损失（如0-1损失，绝对值损失）
        self.optimizer = self.model.get_optimizer(self.opt.lr, self.opt.weight_decay)
        self.compression_scheduler = distiller.CompressionScheduler(self.model)
        self.train_losses = AverageMeter()  # 误差仪表
        self.train_top1 = AverageMeter()  # top1 仪表
        self.train_top5 = AverageMeter()  # top5 仪表
        self.best_precision = 0  # 最好的精确度
        self.start_epoch = 0
        self.train_writer = None
        self.value_writer = None

    def load_data(self):
        test_data = DatasetFromFilename(self.opt.data_root, flag='test')

        train_data = DatasetFromFilename(self.opt.data_root, flag='train')  # 训练集
        val_data = DatasetFromFilename(self.opt.data_root, flag='valid')  # 验证集
        self.test_dataloader = DataLoader(test_data, batch_size=self.opt.batch_size, shuffle=False,
                                          num_workers=self.opt.num_workers)
        self.train_dataloader = DataLoader(train_data, self.opt.batch_size, shuffle=True,
                                           num_workers=self.opt.num_workers)  # 训练集加载器
        self.val_dataloader = DataLoader(val_data, self.opt.batch_size, shuffle=False,
                                         num_workers=self.opt.num_workers)  # 验证集加载器

    def create_write(self):
        if self.opt.vis:
            self.train_writer = SummaryWriter(log_dir='./runs/train_' + datetime.now().strftime('%y%m%d-%H-%M-%S'))
            self.value_writer = SummaryWriter(log_dir='./runs/val_' + datetime.now().strftime('%y%m%d-%H-%M-%S'))

    def train_save_model(self, epoch, val_loss, val_top1, val_top5):
        self.model.save({
            "epoch": epoch + 1,
            "model_name": self.opt.model,
            "state_dict": self.model.state_dict(),
            "best_precision": self.best_precision,
            "optimizer": self.optimizer,
            "valid_loss": [val_loss, val_top1, val_top5],
            'compression_scheduler': self.compression_scheduler.state_dict()
        })  # 保存模型

    def train_load_model(self):
        if self.opt.load_model_path:
            # # 把所有的张量加载到CPU中
            # t.load(opt.load_model_path, map_location=lambda storage, loc: storage)
            # # 把所有的张量加载到GPU 1中
            # t.load(opt.load_model_path, map_location=lambda storage, loc: storage.cuda(1))
            # # 把张量从GPU 1 移动到 GPU 0
            # t.load(opt.load_model_path, map_location={'cuda:1': 'cuda:0'})
            checkpoint = t.load(self.opt.load_model_path)
            self.start_epoch = checkpoint["epoch"]
            # compression_scheduler.load_state_dict(checkpoint['compression_scheduler'], False)
            self.best_precision = checkpoint["best_precision"]
            self.model.load_state_dict(checkpoint["state_dict"])
            self.optimizer = checkpoint['optimizer']
        self.model.to(self.opt.device)  # 加载模型到 GPU

    def load_model(self):
        if self.opt.load_model_path:
            checkpoint = t.load(self.opt.load_model_path)
            self.model.load_state_dict(checkpoint["state_dict"])  # 加载模型
        self.model.to(self.opt.device)

    def save_quantize_model(self):
        if self.opt.quantize_eval:
            self.model.save({
                "model_name": self.opt.model,
                "state_dict": self.model.state_dict(),
                'quantizer_metadata': self.model.quantizer_metadata
            }, './checkpoint/ResNet152_quantize.pth')

    def quantize_model(self):
        if self.opt.quantize_eval:
            self.model.cpu()
            quantizer = quantization.PostTrainLinearQuantizer.from_args(self.model, self.opt)  # 量化模型
            quantizer.prepare_model()
            self.model.to(self.opt.device)

    def load_compress(self):

        if self.opt.compress:
            self.compression_scheduler = distiller.file_config(self.model, self.optimizer, self.opt.compress,
                                                               self.compression_scheduler)  # 加载模型修剪计划表
            self.model.to(self.opt.device)

    def visualization_train(self, input, ii, epoch):
        if self.train_writer:
            grid = make_grid((input.data.cpu() * 0.225 + 0.45).clamp(min=0, max=1))
            self.train_writer.add_image('train_images', grid, ii * (epoch + 1))  # 训练图片
            self.train_writer.add_scalar('loss', self.train_losses.avg, ii * (epoch + 1))  # 训练误差
            self.train_writer.add_text('top1', 'train accuracy top1 %.2f%%' % self.train_top1.avg,
                                       ii * (epoch + 1))  # top1准确率文本
            self.train_writer.add_scalars('accuracy', {'top1': self.train_top1.avg,
                                                       'top5': self.train_top5.avg,
                                                       'loss': self.train_losses.avg}, ii * (epoch + 1))

    def train(self):
        previous_loss = 1e10  # 上次学习的loss
        lr = self.opt.lr
        perf_scores_history = []
        pylogger = PythonLogger(msglogger)
        self.train_load_model()
        self.load_compress()
        self.create_write()
        for epoch in range(self.start_epoch, self.opt.max_epoch):
            self.model.train()
            self.load_data()
            if self.opt.pruning:
                self.compression_scheduler.on_epoch_begin(epoch)  # epoch 开始修剪
            self.train_losses.reset()  # 重置仪表
            self.train_top1.reset()  # 重置仪表
            # print('训练数据集大小', len(train_dataloader))
            total_samples = len(self.train_dataloader.sampler)
            steps_per_epoch = math.ceil(total_samples / self.opt.batch_size)
            train_progressor = ProgressBar(mode="Train  ", epoch=epoch, total_epoch=self.opt.max_epoch,
                                           model_name=self.opt.model,
                                           total=len(self.train_dataloader))
            for ii, (data, labels, img_path) in enumerate(self.train_dataloader):
                if self.opt.pruning:
                    self.compression_scheduler.on_minibatch_begin(epoch, ii, steps_per_epoch,
                                                                  self.optimizer)  # batch 开始修剪
                train_progressor.current = ii + 1  # 训练集当前进度
                # train model
                input = data.to(self.opt.device)
                target = labels.to(self.opt.device)
                score = self.model(input)  # 网络结构返回值
                loss = self.criterion(score, target)  # 计算损失
                if self.opt.pruning:
                    # Before running the backward phase, we allow the scheduler to modify the loss
                    # (e.g. add regularization loss)
                    agg_loss = self.compression_scheduler.before_backward_pass(epoch, ii, steps_per_epoch, loss,
                                                                               optimizer=self.optimizer,
                                                                               return_loss_components=True)  # 模型修建误差
                    loss = agg_loss.overall_loss
                self.train_losses.update(loss.item(), input.size(0))
                # loss = criterion(score[0], target)  # 计算损失   Inception3网络
                self.optimizer.zero_grad()  # 参数梯度设成0
                loss.backward()  # 反向传播
                self.optimizer.step()  # 更新参数

                if opt.pruning:
                    self.compression_scheduler.on_minibatch_end(epoch, ii, steps_per_epoch,
                                                                self.optimizer)  # batch 结束修剪

                precision1_train, precision5_train = accuracy(score, target, topk=(1, 5))  # top1 和 top5 的准确率

                # precision1_train, precision2_train = accuracy(score[0], target, topk=(1, 2))  # Inception3网络
                self.train_losses.update(loss.item(), input.size(0))
                self.train_top1.update(precision1_train[0].item(), input.size(0))
                self.train_top5.update(precision5_train[0].item(), input.size(0))
                train_progressor.current_loss = self.train_losses.avg
                train_progressor.current_top1 = self.train_top1.avg
                train_progressor.current_top5 = self.train_top5.avg

                if (ii + 1) % self.opt.print_freq == 0:
                    self.visualization_train(input, ii, epoch)
                train_progressor()  # 打印进度
            # train_progressor.done()  # 保存训练结果为txt

            print('')
            if self.opt.pruning:
                distiller.log_weights_sparsity(self.model, epoch, loggers=[pylogger])  # 打印模型修剪结果
                self.compression_scheduler.on_epoch_end(epoch, self.optimizer)  # epoch 结束修剪
            val_loss, val_top1, val_top5 = val(self.model, self.criterion, self.val_dataloader, epoch,
                                               self.value_writer)  # 校验模型
            sparsity = distiller.model_sparsity(self.model)
            perf_scores_history.append(distiller.MutableNamedTuple({'sparsity': sparsity, 'top1': val_top1,
                                                                    'top5': val_top5, 'epoch': epoch}))
            # 保持绩效分数历史记录从最好到最差的排序
            # 按稀疏度排序为主排序键，然后按top1、top5、epoch排序
            perf_scores_history.sort(key=operator.attrgetter('sparsity', 'top1', 'top5', 'epoch'), reverse=True)
            for score in perf_scores_history[:1]:
                msglogger.info('==> Best [Top1: %.3f   Top5: %.3f   Sparsity: %.2f on epoch: %d]',
                               score.top1, score.top5, score.sparsity, score.epoch)

            is_best = epoch == perf_scores_history[0].epoch  # 当前epoch 和最佳epoch 一样
            self.best_precision = max(perf_scores_history[0].top1, self.best_precision)  # 最大top1 准确率
            if is_best:
                self.train_save_model(epoch, val_loss, val_top1, val_top5)
            # update learning rate
            # 如果训练误差比上次大　降低学习效率
            if self.train_losses.val > previous_loss:
                lr = lr * self.opt.lr_decay
                # 当loss大于上一次loss,降低学习率
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr

            previous_loss = self.train_losses.val

    def test(self):
        self.load_model()
        self.load_data()
        self.model.eval()  # 把module设成测试模式，对Dropout和BatchNorm有影响
        correct = 0
        total = 0
        msglogger.info('测试数据集大小', len(self.test_dataloader))
        # 量化
        self.quantize_model()
        self.model.eval()  # 把module设成测试模式，对Dropout和BatchNorm有影响
        err_img = [('img_path', 'result', 'label')]
        for ii, (data, labels, img_path) in tqdm(enumerate(self.test_dataloader)):
            input = data.to(self.opt.device)
            labels = labels.to(self.opt.device)
            score = self.model(input)
            # probability = t.nn.functional.softmax(score, dim=1)[:, 1].detach().tolist()  # [:,i] 第i类的权重
            # 将一个K维的任意实数向量压缩（映射）成另一个K维的实数向量，其中向量中的每个元素取值都介于（0，1）之间，并且压缩后的K个值相加等于1(
            # 变成了概率分布)。在选用Softmax做多分类时，可以根据值的大小来进行多分类的任务，如取权重最大的一维
            results = score.max(dim=1)[1].detach()  # max 返回每一行中最大值的那个元素，且返回其索引（返回最大元素在这一行的列索引） 返回最有可能的一类
            # batch_results = [(labels_.item(), self.opt.cate_classes[label_]) for labels_, label_ in zip(labels, label)]
            total += input.size(0)
            correct += (results == labels).sum().item()
            error_list = (results != labels).tolist()
            err_img.extend(
                [(img_path[i], self.opt.cate_classes[results[i]], self.opt.cate_classes[labels[i]]) for i, j in
                 enumerate(error_list) if j == 1])  # 识别错误图片地址,识别标签,正确标签,添加到错误列表

        msglogger.info('Test Accuracy of the model on the {} test images: {} %'.format(total, 100 * correct / total))
        # 错误图片写入csv
        write_err_img(err_img)
        # 保存量化模型
        self.save_quantize_model()

    def recognition(self):
        self.load_model()
        self.model.eval()
        img = image_loader(self.opt.url)
        image = img.view(1, 3, self.opt.image_size, self.opt.image_size).to(self.opt.device)  # 转换image
        outputs = self.model(image)
        result = {}
        for i in range(self.opt.num_classes):  # 计算各分类比重
            result[self.opt.cate_classes[i]] = t.nn.functional.softmax(outputs, dim=1)[:, i].detach().tolist()[0]
            result = sorted(result.items(), key=lambda x: x[1], reverse=True)
        return result

    def sensitivity(self):
        self.load_data()
        self.load_model()
        sensitivities = np.arange(self.opt.sensitivity_range[0], self.opt.sensitivity_range[1],
                                  self.opt.sensitivity_range[2])
        return sensitivity_analysis(self.model, self.criterion, self.test_dataloader, self.opt, sensitivities,
                                    msglogger)


def train(**kwargs):
    train_classifier = Classifier(**kwargs)
    train_classifier.train()


def recognition(**kwargs):
    reco_classifier = Classifier(**kwargs)
    reco_classifier.recognition()


def test(**kwargs):
    test_classifier = Classifier(**kwargs)
    test_classifier.train()


def help():
    """
    打印帮助的信息： python file.py help
    """

    print("""
    usage : python file.py <function> [--args=value]
    <function> := train | test | sensitivity | recognition | help
    example: 
            python {0} train --env='env0701' --lr=0.01
            python {0} test --dataset='path/to/dataset/root/'
            python {0} recognition --url='path/to/dataset/root/' --load_path='prestrain/AlexNet_0121_11-24-50'
            python {0} sensitivity 
            python {0} help
    avaiable args:""".format(__file__))

    from inspect import getsource
    source = (getsource(opt.__class__))
    print(source)


if __name__ == '__main__':
    import fire

    script_dir = os.path.dirname(__file__)
    module_path = os.path.abspath(os.path.join(script_dir, '..', '..'))
    msglogger = apputils.config_pylogger(os.path.join(script_dir, 'logging.conf'), opt.name, opt.output_dir)
    fire.Fire()
