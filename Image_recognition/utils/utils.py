import torch
from config import opt
import csv
import logging
import os
import time


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
    """Configure the Python logger.

    For each execution of the application, we'd like to create a unique log directory.
    By default this directory is named using the date and time of day, so that directories
    can be sorted by recency.  You can also name your experiments and prefix the log
    directory with this name.  This can be useful when accessing experiment data from
    TensorBoard, for example.
    """
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


# 保存模型参数以及优化器参数
def save_checkpoint(state, filename):
    torch.save(state, filename)
    # print("Get Better top1 : %s saving weights to %s" % (state["best_precision"], filename))  # 打印精确度
