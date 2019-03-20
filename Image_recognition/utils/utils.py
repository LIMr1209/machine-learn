import torch
from config import opt


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


# 保存模型参数以及优化器参数
def save_checkpoint(state, filename):
    torch.save(state, filename)
    # print("Get Better top1 : %s saving weights to %s" % (state["best_precision"], filename))  # 打印精确度
