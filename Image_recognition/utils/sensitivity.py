import distiller
from functools import partial
from distiller.data_loggers import *
import torch
from config import opt

from utils.utils import accuracy, AverageMeter


def _validate(data_loader, model, criterion,opt, ):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()

    for validation_step, (inputs, target) in enumerate(data_loader):
        with torch.no_grad():
            inputs, target = inputs.to(opt.device), target.to(opt.device)
            output = model(inputs)

            loss = criterion(output, target)
            precision1, precision2, precision3, precision4, precision5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(precision1[0].item(), input.size(0))
            top5.update(precision5[0].item(), input.size(0))

    return top1.avg, top5.avg, losses.avg


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
        "records": RecordsActivationStatsCollector(model, classes=[torch.nn.Conv2d])
    })

    return {k: (genCollectors() if k in phases else missingdict())
            for k in ('train', 'valid', 'test')}


# 模型敏感性分析
def test(test_loader, model, criterion, loggers, activations_collectors, args):
    """Model Test"""
    if activations_collectors is None:
        activations_collectors = create_activation_stats_collectors(model, None)
    with collectors_context(activations_collectors["test"]) as collectors:
        top1, top5, lossses = _validate(test_loader, model, criterion, opt)
        distiller.log_activation_statsitics(-1, "test", loggers, collector=collectors['sparsity'])
    return top1, top5, lossses


def sensitivity_analysis(model, criterion, data_loader, opt, sparsities):
    # 可以调用此示例应用程序来对模型执行敏感性分析。输出保存到csv和png。
    test_fnc = partial(test, test_loader=data_loader, criterion=criterion, args=opt,
                       activations_collectors=create_activation_stats_collectors(model))
    which_params = [param_name for param_name, _ in model.named_parameters()]
    sensitivity = distiller.perform_sensitivity_analysis(model,
                                                         net_params=which_params,
                                                         sparsities=sparsities,
                                                         test_func=test_fnc,
                                                         group=opt.sensitivity)
    distiller.sensitivities_to_png(sensitivity, 'sensitivity.png')
    distiller.sensitivities_to_csv(sensitivity, 'sensitivity.csv')
