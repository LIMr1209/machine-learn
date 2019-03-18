#
# Copyright (c) 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import argparse

import distiller
import distiller.quantization
import examples.automated_deep_compression as adc
from distiller.utils import float_range_argparse_checker as float_range
import distiller.models as models

SUMMARY_CHOICES = ['sparsity', 'compute', 'model', 'modules', 'png', 'png_w_params', 'onnx']


def get_parser():
    parser = argparse.ArgumentParser(description='Distiller image classification model compression')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet152', type=lambda s: s.lower(),
                        choices=models.ALL_MODEL_NAMES,
                        help='model architecture: ' +
                             ' | '.join(models.ALL_MODEL_NAMES) +
                             ' (default: resnet152)')
    parser.add_argument('-j', '--workers', default=44, type=int, metavar='N',
                        help='number of data loading workers (default: 44)')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=16, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=0e-5, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--print-freq', '-p', default=500, type=int,
                        metavar='N', help='print frequency (default: 50)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--activation-stats', '--act-stats', nargs='+', metavar='PHASE', default=list(),
                        help='collect activation statistics on phases: train, valid, and/or test'
                             ' (WARNING: this slows down training)')  # 收集阶段的激活统计信息：列车、有效和/或测试（警告：这会减慢训练速度）
    parser.add_argument('--masks-sparsity', dest='masks_sparsity', action='store_true', default=False,
                        help='print masks sparsity table at end of each epoch')  # 打印掩盖稀疏表 在end of each epoch
    parser.add_argument('--param-hist', dest='log_params_histograms', action='store_true', default=False,
                        help='log the parameter tensors histograms to file (WARNING: this can use significant disk space)')  # 将参数张量柱状图记录到文件中（警告：这可能会占用大量磁盘空间）
    parser.add_argument('--summary', type=lambda s: s.lower(), choices=SUMMARY_CHOICES,
                        help='print a summary of the model, and exit - options: ' +
                             ' | '.join(SUMMARY_CHOICES))  # 打印模型摘要，并退出-选项
    parser.add_argument('--compress', dest='compress', type=str, nargs='?', action='store',
                        help='configuration file for pruning the model (default is to use hard-coded schedule)')  # 用于修剪模型的配置文件（默认为使用硬编码计划）
    parser.add_argument('--sense', dest='sensitivity', choices=['element', 'filter', 'channel'],
                        type=lambda s: s.lower(),
                        help='test the thinnify of layers to pruning')  # 测试修剪层的敏感性
    parser.add_argument('--sense-range', dest='sensitivity_range', type=float, nargs=3, default=[0.4, 0.9, 0.1],
                        help='an optional parameter for sensitivity testing providing the range of sparsities to test.\n'
                             'This is equivalent to creating sensitivities = np.arange(start, stop, step)')  # 灵敏度测试的可选参数，提供要测试的稀疏度范围。这相当于创建灵敏度=np.arange（开始、停止、步骤）
    parser.add_argument('--extras', default=None, type=str,
                        help='file with extra configuration information')  # 包含额外配置信息的文件
    parser.add_argument('--deterministic', '--det', action='store_true',
                        help='Ensure deterministic execution for re-producible results.')  # 确保可重复生产结果的确定性执行。
    parser.add_argument('--gpus', metavar='DEV_ID', default=None,
                        help='Comma-separated list of GPU device IDs to be used (default is to use all available devices)')  # 要使用的GPU设备ID的逗号分隔列表（默认为使用所有可用设备）
    parser.add_argument('--cpu', action='store_true', default=False,
                        help='Use CPU only. \n'
                             'Flag not set => uses GPUs according to the --gpus flag value.'
                             'Flag set => overrides the --gpus flag')
    parser.add_argument('--name', '-n', metavar='NAME', default=None, help='Experiment name')  # 实验名
    parser.add_argument('--out-dir', '-o', dest='output_dir', default='logs',
                        help='Path to dump logs and checkpoints')  # 转储日志和检查点的路径
    parser.add_argument('--validation-split', '--valid-size', '--vs', dest='validation_split',
                        type=float_range(exc_max=True), default=0.1,
                        help='Portion of training dataset to set aside for validation')  # 用于验证的培训数据集的一部分
    parser.add_argument('--effective-train-size', '--etrs', type=float_range(exc_min=True), default=1.,
                        help='Portion of training dataset to be used in each epoch. '
                             'NOTE: If --validation-split is set, then the value of this argument is applied '
                             'AFTER the train-validation split according to that argument')
    # 培训数据集的一部分将在每个时期使用。
    # '注意：如果设置了--validation split，则应用此参数的值'
    # '在根据该参数划分列车验证之后'
    parser.add_argument('--effective-valid-size', '--evs', type=float_range(exc_min=True), default=1.,
                        help='Portion of validation dataset to be used in each epoch. '
                             'NOTE: If --validation-split is set, then the value of this argument is applied '
                             'AFTER the train-validation split according to that argument')
    # 在每个时期使用的验证数据集的一部分。
    # '注意：如果设置了--validation split，则应用此参数的值'
    # '根据该参数，在列车验证拆分后
    parser.add_argument('--effective-test-size', '--etes', type=float_range(exc_min=True), default=1.,
                        help='Portion of test dataset to be used in each epoch')  # 每个时期使用的测试数据集的一部分
    parser.add_argument('--confusion', dest='display_confusion', default=False, action='store_true',
                        help='Display the confusion matrix')  # 显示混淆矩阵
    parser.add_argument('--earlyexit_lossweights', type=float, nargs='*', dest='earlyexit_lossweights', default=None,
                        help='List of loss weights for early exits (e.g. --earlyexit_lossweights 0.1 0.3)')  # 提前退出的的损失权重列表（例如0.1 0.3）
    parser.add_argument('--earlyexit_thresholds', type=float, nargs='*', dest='earlyexit_thresholds', default=None,
                        help='List of EarlyExit thresholds (e.g. --earlyexit_thresholds 1.2 0.9)')  # 提前退出阈值列表（例如1.2 0.9）
    parser.add_argument('--num-best-scores', dest='num_best_scores', default=1, type=int,
                        help='number of best scores to track and report (default: 1)')  # 要跟踪和报告的最佳分数数（默认值：1）
    parser.add_argument('--load-serialized', dest='load_serialized', action='store_true', default=False,
                        help='Load a model without DataParallel wrapping it')  # 加载模型时不使用数据并行包装
    parser.add_argument('--thinnify', dest='thinnify', action='store_true', default=False,
                        help='physically remove zero-filters and create a smaller model')  # 物理删除零过滤器并创建较小的模型

    distiller.knowledge_distillation.add_distillation_args(parser, models.ALL_MODEL_NAMES, True)
    distiller.quantization.add_post_train_quant_args(parser)
    distiller.pruning.greedy_filter_pruning.add_greedy_pruner_args(parser)
    adc.automl_args.add_automl_args(parser)
    return parser
