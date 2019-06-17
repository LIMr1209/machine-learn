import torch
from torch.nn import init
from torch.optim import lr_scheduler
from config import opt
import csv
import logging
import os
import time


# 检查数据
def check_date(img_path, tag, msglogger):
    for i in range(len(tag)):
        if tag[i] not in img_path[i]:
            msglogger.info('数据集错误')
            return False
    return True


# 仪表盘
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def write_err_img(err_img):
    with open(opt.error_img, 'w', newline='') as f:
        csv_write = csv.writer(f)
        for err in err_img:
            csv_write.writerow(err)


# 准确率
def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        # https: // pytorch - cn.readthedocs.io / zh / latest / package_references / torch /  # torchtopk
        _, pred = output.topk(maxk, 1, True, True)  # 返回分值最大的两类的index
        pred = pred.t()  # 转置
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


# log日志
def config_pylogger(log_cfg_file, experiment_name, output_dir='logs'):
    timestr = time.strftime("%Y.%m.%d-%H%M%S")
    exp_full_name = timestr if experiment_name is None else experiment_name + '___' + timestr
    logdir = os.path.join(output_dir, exp_full_name)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    log_filename = os.path.join(logdir, exp_full_name + '.log')
    if os.path.isfile(log_cfg_file):
        logging.config.fileConfig(log_cfg_file, defaults={'logfilename': log_filename})
    msglogger = logging.getLogger()
    msglogger.logdir = logdir
    msglogger.log_filename = log_filename
    msglogger.info('Log file for this run: ' + os.path.realpath(log_filename))
    return msglogger


#  学习速率调度器
def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'plateau':
        # 当网络的评价指标不在提升的时候，可以通过降低网络的学习率来提高网络性能。
        # 当train_loss 不在下降时 降低学习率
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=opt.lr_gamma, patience=4, verbose=True)
    elif opt.lr_policy == 'step':
        # 当epoch in [3,6,9,..]  学习率下降
        scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=opt.lr_gamma)
    elif opt.lr_policy == 'multi':
        # 当epoch in milestones  学习率下降
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=opt.lr_epoch, gamma=opt.lr_gamma)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


# 权重初始化
def init_net(net, init_type='normal', init_gain=0.02):
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


# 保存模型参数以及优化器参数
def save_checkpoint(state, filename):
    torch.save(state, filename)
    # print("Get Better top1 : %s saving weights to %s" % (state["best_precision"], filename))  # 打印精确度
