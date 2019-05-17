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


def test(**kwargs):
    with t.no_grad():
        opt._parse(kwargs)
        # configure model
        model = getattr(models, opt.model)()
        if opt.load_model_path:
            # model = t.load(opt.load_model_path)
            checkpoint = t.load(opt.load_model_path)
            model.load_state_dict(checkpoint['state_dict'])  # 加载模型
        model.to(opt.device)
        model.eval()  # 把module设成测试模式，对Dropout和BatchNorm有影响
        # data
        test_data = DatasetFromFilename(opt.data_root, flag='train')  # 测试集
        test_dataloader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
        correct = 0
        total = 0
        msglogger.info('测试数据集大小', len(test_dataloader))
        # 量化
        if opt.quantize_eval:
            model.cpu()
            quantizer = quantization.PostTrainLinearQuantizer.from_args(model, opt)  # 量化模型
            quantizer.prepare_model()
            model.to(opt.device)
        model.eval()  # 把module设成测试模式，对Dropout和BatchNorm有影响
        err_img = [('img_path', 'result', 'label')]
        for ii, (data, labels, img_path) in tqdm(enumerate(test_dataloader)):
            input = data.to(opt.device)
            labels = labels.to(opt.device)
            score = model(input)
            # probability = t.nn.functional.softmax(score, dim=1)[:, 1].detach().tolist()  # [:,i] 第i类的权重
            # 将一个K维的任意实数向量压缩（映射）成另一个K维的实数向量，其中向量中的每个元素取值都介于（0，1）之间，并且压缩后的K个值相加等于1(
            # 变成了概率分布)。在选用Softmax做多分类时，可以根据值的大小来进行多分类的任务，如取权重最大的一维
            results = score.max(dim=1)[1].detach()  # max 返回每一行中最大值的那个元素，且返回其索引（返回最大元素在这一行的列索引） 返回最有可能的一类
            # batch_results = [(labels_.item(), opt.cate_classes[label_]) for labels_, label_ in zip(labels, label)]
            total += input.size(0)
            correct += (results == labels).sum().item()
            error_list = (results != labels).tolist()
            err_img.extend([(img_path[i], opt.cate_classes[results[i]], opt.cate_classes[labels[i]]) for i, j in
                            enumerate(error_list) if j == 1])  # 识别错误图片地址,识别标签,正确标签,添加到错误列表

        msglogger.info('Test Accuracy of the model on the {} test images: {} %'.format(total, 100 * correct / total))
        # 错误图片写入csv
        write_err_img(err_img)
        # 保存量化模型
        if opt.quantize_eval:
            model.save({
                # "model_name": opt.model,
                "state_dict": model.state_dict(),
                'quantizer_metadata': model.quantizer_metadata
            }, './checkpoint/ResNet152_quantize.pth')
            t.save(model.models, './checkpoint/ResNet152_quantize1.pth')


def recognition(**kwargs):
    with t.no_grad():  # 用来标志计算要被计算图隔离出去
        opt._parse(kwargs)
        image = image_loader(opt.url)
        model = getattr(models, opt.model)()
        if opt.load_model_path:
            checkpoint = t.load(opt.load_model_path)
            model.load_state_dict(checkpoint["state_dict"])  # 加载模型
        model.to(opt.device)
        model.eval()
        image = image.view(1, 3, opt.image_size, opt.image_size).to(opt.device)  # 转换image
        outputs = model(image)
        result = {}
        for i in range(opt.num_classes):  # 计算各分类比重
            result[opt.cate_classes[i]] = t.nn.functional.softmax(outputs, dim=1)[:, i].detach().tolist()[0]
            result = sorted(result.items(), key=lambda x: x[1], reverse=True)
        print(result)


def train(**kwargs):
    train_writer = None
    value_writer = None
    if opt.vis:
        train_writer = SummaryWriter(log_dir='./runs/train_' + datetime.now().strftime('%y%m%d-%H-%M-%S'))
        value_writer = SummaryWriter(log_dir='./runs/val_' + datetime.now().strftime('%y%m%d-%H-%M-%S'))
    opt._parse(kwargs)
    previous_loss = 1e10  # 上次学习的loss
    best_precision = 0  # 最好的精确度
    start_epoch = 0
    lr = opt.lr
    perf_scores_history = []
    # step1: criterion and optimizer
    # 1. 铰链损失（Hinge Loss）：主要用于支持向量机（SVM） 中；
    # 2. 互熵损失 （Cross Entropy Loss，Softmax Loss ）：用于Logistic 回归与Softmax 分类中；
    # 3. 平方损失（Square Loss）：主要是最小二乘法（OLS）中；
    # 4. 指数损失（Exponential Loss） ：主要用于Adaboost 集成学习算法中；
    # 5. 其他损失（如0-1损失，绝对值损失）
    criterion = t.nn.CrossEntropyLoss().to(opt.device)  # 损失函数
    # step2: meters
    train_losses = AverageMeter()  # 误差仪表
    train_top1 = AverageMeter()  # top1 仪表
    train_top5 = AverageMeter()  # top5 仪表
    pylogger = PythonLogger(msglogger)
    # step3: configure model
    model = getattr(models, opt.model)()  # 获得网络结构
    compression_scheduler = distiller.CompressionScheduler(model)
    optimizer = model.get_optimizer(opt.lr, opt.weight_decay)  # 优化器
    if opt.load_model_path:
        # # 把所有的张量加载到CPU中
        # t.load(opt.load_model_path, map_location=lambda storage, loc: storage)
        # # 把所有的张量加载到GPU 1中
        # t.load(opt.load_model_path, map_location=lambda storage, loc: storage.cuda(1))
        # # 把张量从GPU 1 移动到 GPU 0
        # t.load(opt.load_model_path, map_location={'cuda:1': 'cuda:0'})
        checkpoint = t.load(opt.load_model_path)
        start_epoch = checkpoint["epoch"]
        # compression_scheduler.load_state_dict(checkpoint['compression_scheduler'], False)
        best_precision = checkpoint["best_precision"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer = checkpoint['optimizer']
        lr = optimizer.param_groups[0]['lr']
    model.to(opt.device)  # 加载模型到 GPU

    if opt.compress:
        compression_scheduler = distiller.file_config(model, optimizer, opt.compress,
                                                      compression_scheduler)  # 加载模型修剪计划表
        model.to(opt.device)
    # train
    for epoch in range(start_epoch, opt.max_epoch):
        model.train()
        # step4: data_image
        train_data = DatasetFromFilename(opt.data_root, flag='train')  # 训练集
        val_data = DatasetFromFilename(opt.data_root, flag='valid')  # 验证集
        train_dataloader = DataLoader(train_data, opt.batch_size, shuffle=True, num_workers=opt.num_workers)  # 训练集加载器
        val_dataloader = DataLoader(val_data, opt.batch_size, shuffle=False, num_workers=opt.num_workers)  # 验证集加载器
        if opt.pruning:
            compression_scheduler.on_epoch_begin(epoch)  # epoch 开始修剪
        train_losses.reset()  # 重置仪表
        train_top1.reset()  # 重置仪表
        # print('训练数据集大小', len(train_dataloader))
        total_samples = len(train_dataloader.sampler)
        steps_per_epoch = math.ceil(total_samples / opt.batch_size)
        train_progressor = ProgressBar(mode="Train  ", epoch=epoch, total_epoch=opt.max_epoch,
                                       model_name=opt.model, lr=lr,
                                       total=len(train_dataloader))
        for ii, (data, labels, img_path) in enumerate(train_dataloader):

            if opt.pruning:
                compression_scheduler.on_minibatch_begin(epoch, ii, steps_per_epoch, optimizer)  # batch 开始修剪
            train_progressor.current = ii + 1  # 训练集当前进度
            # train model
            input = data.to(opt.device)
            target = labels.to(opt.device)
            if train_writer:
                grid = make_grid((input.data.cpu() * 0.225 + 0.45).clamp(min=0, max=1))
                train_writer.add_image('train_images', grid, ii * (epoch + 1))  # 训练图片
            score = model(input)  # 网络结构返回值
            # 计算损失
            loss = criterion(score, target)
            if opt.pruning:
                # Before running the backward phase, we allow the scheduler to modify the loss
                # (e.g. add regularization loss)
                agg_loss = compression_scheduler.before_backward_pass(epoch, ii, steps_per_epoch, loss,
                                                                      optimizer=optimizer,
                                                                      return_loss_components=True)  # 模型修建误差
                loss = agg_loss.overall_loss
            train_losses.update(loss.item(), input.size(0))
            # loss = criterion(score[0], target)  # 计算损失   Inception3网络
            optimizer.zero_grad()  # 参数梯度设成0
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            if opt.pruning:
                compression_scheduler.on_minibatch_end(epoch, ii, steps_per_epoch, optimizer)  # batch 结束修剪

            precision1_train, precision5_train = accuracy(score, target, topk=(1, 5))  # top1 和 top5 的准确率

            # writer.add_graph(model, input)
            # precision1_train, precision2_train = accuracy(score[0], target, topk=(1, 2))  # Inception3网络
            train_losses.update(loss.item(), input.size(0))
            train_top1.update(precision1_train[0].item(), input.size(0))
            train_top5.update(precision5_train[0].item(), input.size(0))
            train_progressor.current_loss = train_losses.avg
            train_progressor.current_top1 = train_top1.avg
            train_progressor.current_top5 = train_top5.avg
            train_progressor()  # 打印进度
            if ii % opt.print_freq:
                if train_writer:
                    train_writer.add_scalar('loss', train_losses.avg, ii * (epoch + 1))  # 训练误差
                    train_writer.add_text('top1', 'train accuracy top1 %s' % train_top1.avg,
                                          ii * (epoch + 1))  # top1准确率文本
                    train_writer.add_scalars('accuracy', {'top1': train_top1.avg,
                                                          'top5': train_top5.avg,
                                                          'loss': train_losses.avg}, ii * (epoch + 1))
        # train_progressor.done()  # 保存训练结果为txt
        # validate and visualize
        print('')
        if opt.pruning:
            distiller.log_weights_sparsity(model, epoch, loggers=[pylogger])  # 打印模型修剪结果
            compression_scheduler.on_epoch_end(epoch, optimizer)  # epoch 结束修剪
        val_loss, val_top1, val_top5 = val(model, criterion, val_dataloader, epoch, value_writer,lr)  # 校验模型
        sparsity = distiller.model_sparsity(model)
        perf_scores_history.append(distiller.MutableNamedTuple({'sparsity': sparsity, 'top1': val_top1,
                                                                'top5': val_top5, 'epoch': epoch + 1, 'lr': lr}, ))
        # 保持绩效分数历史记录从最好到最差的排序
        # 按稀疏度排序为主排序键，然后按top1、top5、epoch排序
        perf_scores_history.sort(key=operator.attrgetter('sparsity', 'top1', 'top5', 'epoch'), reverse=True)
        for score in perf_scores_history[:1]:
            msglogger.info('==> Best [Top1: %.3f   Top5: %.3f   Sparsity: %.2f on epoch: %d lr: %d]',
                           score.top1, score.top5, score.sparsity, score.epoch, lr)

        is_best = epoch == perf_scores_history[0].epoch  # 当前epoch 和最佳epoch 一样
        best_precision = max(perf_scores_history[0].top1, best_precision)  # 最大top1 准确率
        if is_best:
            model.save({
                "epoch": epoch + 1,
                "model_name": opt.model,
                "state_dict": model.state_dict(),
                "best_precision": best_precision,
                "optimizer": optimizer,
                "valid_loss": [val_loss, val_top1, val_top5],
                'compression_scheduler': compression_scheduler.state_dict()
            })  # 保存模型
        # update learning rate
        # 如果训练误差比上次大　降低学习效率
        if train_losses.val > previous_loss:
            lr = lr * opt.lr_decay
            # 当loss大于上一次loss,降低学习率
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        previous_loss = train_losses.val


# 模型敏感性分析
def sensitivity(**kwargs):
    opt._parse(kwargs)
    test_data = DatasetFromFilename(opt.data_root, flag='test')
    test_dataloader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    criterion = t.nn.CrossEntropyLoss().to(opt.device)  # 损失函数
    model = getattr(models, opt.model)()
    if opt.load_model_path:
        checkpoint = t.load(opt.load_model_path)
        model.load_state_dict(checkpoint["state_dict"])
    model.to(opt.device)
    sensitivities = np.arange(opt.sensitivity_range[0], opt.sensitivity_range[1], opt.sensitivity_range[2])
    return sensitivity_analysis(model, criterion, test_dataloader, opt, sensitivities, msglogger)


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
