from config import opt
import torch as t
import models
from data.dataset import DatasetFromFilename
from torch.utils.data import DataLoader
# from utils.csv import write_csv
from utils.image_loader import image_loader
from utils.utils import AverageMeter, accuracy
from utils.visualize import Visualizer
from utils.progress_bar import ProgressBar
from tqdm import tqdm


# torch.cuda.manual_seed(seed)  #随机数种子,当使用随机数时,关闭进程后再次生成和上次得一样

def test(**kwargs):
    with t.no_grad():
        opt._parse(kwargs)
        # configure model
        model = getattr(models, opt.model)()
        if opt.load_model_path:
            checkpoint = t.load(opt.load_model_path)
            model.load_state_dict(checkpoint["state_dict"])  # 加载模型
        model.to(opt.device)
        model.eval()  # 把module设成测试模式，对Dropout和BatchNorm有影响
        # data
        test_data = DatasetFromFilename(opt.data_root, test=True)  # 测试集
        test_dataloader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
        correct = 0
        total = 0
        print('测试数据集大小', len(test_dataloader))
        for ii, (data, labels) in tqdm(enumerate(test_dataloader)):
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

        print('Test Accuracy of the model on the {} test images: {} %'.format(total, 100 * correct / total))


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
        print(result)


def train(**kwargs):
    opt._parse(kwargs)
    if opt.vis:
        vis = Visualizer(opt.env, port=opt.vis_port)  # 开启visdom 可视化
    previous_loss = 1e10  # 上次学习的loss
    best_precision = 0  # 最好的精确度
    start_epoch = 0
    lr = opt.lr
    # step1: criterion and optimizer
    # 1. 铰链损失（Hinge Loss）：主要用于支持向量机（SVM） 中；
    # 2. 互熵损失 （Cross Entropy Loss，Softmax Loss ）：用于Logistic 回归与Softmax 分类中；
    # 3. 平方损失（Square Loss）：主要是最小二乘法（OLS）中；
    # 4. 指数损失（Exponential Loss） ：主要用于Adaboost 集成学习算法中；
    # 5. 其他损失（如0-1损失，绝对值损失）
    criterion = t.nn.CrossEntropyLoss().to(opt.device)  # 损失函数
    # step2: meters
    train_losses = AverageMeter()
    train_top1 = AverageMeter()
    # step3: configure model
    model = getattr(models, opt.model)()
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
        best_precision = checkpoint["best_precision"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    model.to(opt.device)
    # step4: data
    train_data = DatasetFromFilename(opt.data_root, train=True)  # 训练集
    val_data = DatasetFromFilename(opt.data_root, train=False)  # 验证集
    train_dataloader = DataLoader(train_data, opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    val_dataloader = DataLoader(val_data, opt.batch_size, shuffle=False, num_workers=opt.num_workers)

    # train
    for epoch in range(start_epoch, opt.max_epoch):
        model.train()
        train_losses.reset()  # 重置仪表
        train_top1.reset()  # 重置仪表
        # print('训练数据集大小', len(train_dataloader))
        train_progressor = ProgressBar(mode="Train  ", epoch=epoch, total_epoch=opt.max_epoch,
                                       model_name=opt.model,
                                       total=len(train_dataloader))
        for ii, (data, labels) in enumerate(train_dataloader):
            train_progressor.current = ii+1
            # train model
            input = data.to(opt.device)
            target = labels.to(opt.device)

            score = model(input)
            loss = criterion(score, target)  # 计算损失
            # loss = criterion(score[0], target)  # 计算损失   Inception3网络
            optimizer.zero_grad()  # 参数梯度设成0
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            # meters update and visualize
            precision1_train, precision2_train = accuracy(score, target, topk=(1, 2))  # top1 和 top2 的准确率
            # precision1_train, precision2_train = accuracy(score[0], target, topk=(1, 2))  # Inception3网络
            train_losses.update(loss.item(), input.size(0))
            train_top1.update(precision1_train[0].item(), input.size(0))
            train_progressor.current_loss = train_losses.avg
            train_progressor.current_top1 = train_top1.avg
            if (ii + 1) % opt.print_freq == 0:
                if opt.vis:
                    vis.plot('loss', train_losses.val)  # 绘图
                else:
                    print('loss', train_losses.val)
            train_progressor()
        # train_progressor.done()  # 保存训练结果为txt
        # validate and visualize
        print('')
        valid_loss = val(model, epoch, criterion, val_dataloader)  # 校验模型
        is_best = valid_loss[1] > best_precision  # 精确度比较，如果此次比上次大　　保存模型
        best_precision = max(valid_loss[1], best_precision)
        if is_best:
            model.save({
                "epoch": epoch + 1,
                "model_name": opt.model,
                "state_dict": model.state_dict(),
                "best_precision": best_precision,
                "optimizer": optimizer.state_dict(),
                "valid_loss": valid_loss,
            })  # 保存模型
        # update learning rate
        # 如果训练误差比上次大　降低学习效率
        if train_losses.val > previous_loss:
            lr = lr * opt.lr_decay
            # 当loss大于上一次loss,降低学习率
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        previous_loss = train_losses.val


def val(model, epoch, criterion, dataloader):
    with t.no_grad():
        """
        计算模型在验证集上的准确率等信息
        """
        model.eval()
        losses = AverageMeter()
        top1 = AverageMeter()
        # print('验证数据集大小', len(dataloader))
        val_progressor = ProgressBar(mode="Val  ", epoch=epoch, total_epoch=opt.max_epoch, model_name=opt.model,
                                     total=len(dataloader))
        for ii, (data, labels) in enumerate(dataloader):
            val_progressor.current = ii+1
            input = data.to(opt.device)
            labels = labels.to(opt.device)
            score = model(input)
            loss = criterion(score, labels)

            # 2.2.2 measure accuracy and record loss
            precision1, precision2 = accuracy(score, labels, topk=(1, 2))  # top1 和 top2 的准确率
            losses.update(loss.item(), input.size(0))
            top1.update(precision1[0].item(), input.size(0))
            val_progressor.current_loss = losses.avg
            val_progressor.current_top1 = top1.avg
            val_progressor()
        print('')
        # val_progressor.done() # 保存校验结果为txt
        return [losses.avg, top1.avg]


def help():
    """
    打印帮助的信息： python file.py help
    """

    print("""
    usage : python file.py <function> [--args=value]
    <function> := train | test | help
    example: 
            python {0} train --env='env0701' --lr=0.01
            python {0} test --dataset='path/to/dataset/root/'
            python {0} recognition --url='path/to/dataset/root/' --load_path='prestrain/AlexNet_0121_11-24-50'
            python {0} help
    avaiable args:""".format(__file__))

    from inspect import getsource
    source = (getsource(opt.__class__))
    print(source)


if __name__ == '__main__':
    import fire

    fire.Fire()
