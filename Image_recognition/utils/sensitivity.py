import distiller
from functools import partial
from distiller.data_loggers import *
import torch as t
from torchvision.utils import make_grid

from config import opt
from utils.progress_bar import ProgressBar
from utils.utils import accuracy, AverageMeter


def val(model, criterion, dataloader, epoch=None, val_writer=None, msglogger=None):
    with t.no_grad():
        """
        计算模型在验证集上的准确率等信息
        """
        model.eval()
        val_losses = AverageMeter()
        val_top1 = AverageMeter()
        val_top5 = AverageMeter()
        val_progressor = None
        if not msglogger:
            val_progressor = ProgressBar(mode="Val  ", epoch=epoch, total_epoch=opt.max_epoch, model_name=opt.model,
                                         total=len(dataloader))
        for ii, (data, labels, img_path) in enumerate(dataloader):
            input = data.to(opt.device)
            labels = labels.to(opt.device)
            score = model(input)
            if val_writer:
                grid = make_grid((input.data.cpu() * 0.225 + 0.45).clamp(min=0, max=1))
                val_writer.add_image('train_images', grid, ii * (epoch + 1))  # 测试图片
            loss = criterion(score, labels)

            precision1, precision5 = accuracy(score, labels, topk=(1, 5))  # top1 和 top2 的准确率
            val_losses.update(loss.item(), input.size(0))
            val_top1.update(precision1[0].item(), input.size(0))
            val_top5.update(precision5[0].item(), input.size(0))
            if val_progressor:
                val_progressor.current = ii + 1
                val_progressor.current_loss = val_losses.avg
                val_progressor.current_top1 = val_top1.avg
                val_progressor.current_top5 = val_top5.avg
                if val_writer:
                    val_writer.add_scalar('train_loss', val_losses.avg, ii * (epoch + 1))  # 训练误差
                    val_writer.add_text('train_top1', 'train accuracy top1 %s%%' % val_top1.avg,
                                        ii * (epoch + 1))  # top1准确率文本
                    val_writer.add_text('train_top5', 'train accuracy top5 %s%%' % val_top5.avg,
                                        ii * (epoch + 1))  # top5准确率文本
                    val_writer.add_pr_curve('val_acc', val_top1.avg, ii * (epoch + 1))  # 精确率
                val_progressor()
        if msglogger:
            msglogger.info('==> Top1: %.3f    Top5: %.3f    Loss: %.3f\n',
                           val_top1.avg, val_top5.avg, val_losses.avg)
        if val_progressor:
            print('')
        return [val_losses.avg, val_top1.avg, val_top5.avg]


class missingdict(dict):
    """This is a little trick to prevent KeyError"""

    def __missing__(self, key):
        return None  # note, does *not* set self[key] - we don't want defaultdict's behavior


def create_activation_stats_collectors(model, *phases):
    distiller.utils.assign_layer_fq_names(model)

    genCollectors = lambda: missingdict({
        "sparsity": SummaryActivationStatsCollector(model, "sparsity",
                                                    lambda t: 100 * distiller.utils.sparsity(t)),
        "l1_channels": SummaryActivationStatsCollector(model, "l1_channels",
                                                       distiller.utils.activation_channels_l1),
        "apoz_channels": SummaryActivationStatsCollector(model, "apoz_channels",
                                                         distiller.utils.activation_channels_apoz),
        "mean_channels": SummaryActivationStatsCollector(model, "mean_channels",
                                                         distiller.utils.activation_channels_means),
        "records": RecordsActivationStatsCollector(model, classes=[t.nn.Conv2d])
    })

    return {k: (genCollectors() if k in phases else missingdict())
            for k in ('train', 'valid', 'test')}


# 模型敏感性分析
def test(test_loader, model, criterion, loggers, activations_collectors, msglogger, args):
    """Model Test"""
    if activations_collectors is None:
        activations_collectors = create_activation_stats_collectors(model, None)
    with collectors_context(activations_collectors["test"]) as collectors:
        lossses, top1, top5, = val(model, criterion, test_loader, msglogger=msglogger)
        distiller.log_activation_statsitics(-1, "test", loggers, collector=collectors['sparsity'])
    return top1, top5, lossses


def sensitivity_analysis(model, criterion, data_loader, opt, sparsities, msglogger):
    msglogger.info("Running sensitivity tests")
    pylogger = PythonLogger(msglogger)
    if not isinstance(pylogger, list):
        pylogger = [pylogger]
    # 可以调用此示例应用程序来对模型执行敏感性分析。输出保存到csv和png。
    test_fnc = partial(test, test_loader=data_loader, criterion=criterion, loggers=pylogger, args=opt,
                       msglogger=msglogger,
                       activations_collectors=create_activation_stats_collectors(model))
    which_params = [param_name for param_name, _ in model.named_parameters()]
    sensitivity = distiller.perform_sensitivity_analysis(model,
                                                         net_params=which_params,
                                                         sparsities=sparsities,
                                                         test_func=test_fnc,
                                                         group=opt.sensitivity)
    distiller.sensitivities_to_png(sensitivity, 'sensitivity.png')
    distiller.sensitivities_to_csv(sensitivity, 'sensitivity.csv')
