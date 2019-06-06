import argparse
import template

parser = argparse.ArgumentParser(description='EDSR and MDSR')

parser.add_argument('--debug', action='store_true',
                    help='Enables debug mode')
parser.add_argument('--template', default='.',
                    help='You can set various templates in option.py')

# Hardware specifications # 硬件规格
parser.add_argument('--n_threads', type=int, default=6,
                    help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true',
                    help='use cpu only')
parser.add_argument('--n_GPUs', type=int, default=1,
                    help='number of GPUs')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')

# Data specifications  #数据规格
parser.add_argument('--dir_data', type=str, default='/image',
                    help='dataset directory')  # 数据集目录
parser.add_argument('--dir_demo', type=str, default='../test',
                    help='demo image directory')  # 演示图像目录
parser.add_argument('--data_train', type=str, default='DIV2K',
                    help='train dataset name')  # 训练数据集名称
parser.add_argument('--data_test', type=str, default='DIV2K',
                    help='test dataset name')  # 测试数据集名称
parser.add_argument('--data_range', type=str, default='1-800/801-810',
                    help='train/test data range')  # 训练/测试数据范围
parser.add_argument('--ext', type=str, default='sep',
                    help='dataset file extension')  # 数据集文件扩展名
parser.add_argument('--scale', type=str, default='4',
                    help='super resolution scale')  # 超分辨率标度
parser.add_argument('--patch_size', type=int, default=192,
                    help='output patch size')  # 输出补丁大小
parser.add_argument('--rgb_range', type=int, default=255,
                    help='maximum value of RGB')  # RGB的最大值
parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use')  # 要使用的颜色通道数
parser.add_argument('--chop', action='store_true',
                    help='enable memory-efficient forward')  # 启用内存高效转发
parser.add_argument('--no_augment', action='store_true',
                    help='do not use data augmentation')  # 不使用数据扩充

# Model specifications  # 模型规格
parser.add_argument('--model', default='EDSR',
                    help='model name')  # 模型名

parser.add_argument('--act', type=str, default='relu',
                    help='activation function')  # 激活函数
parser.add_argument('--pre_train', type=str, default='',
                    help='pre-trained model directory')  # 预先培训的模型文件
parser.add_argument('--extend', type=str, default='.',
                    help='pre-trained model directory')  # 预先培训的模型目录
parser.add_argument('--n_resblocks', type=int, default=16,
                    help='number of residual blocks')  # 剩余块数
parser.add_argument('--n_feats', type=int, default=64,
                    help='number of feature maps')  # 特征映射数量
parser.add_argument('--res_scale', type=float, default=1,
                    help='residual scaling')  # 残留结垢
parser.add_argument('--shift_mean', default=True,
                    help='subtract pixel mean from the input')  # 从输入中减去像素平均值
parser.add_argument('--dilation', action='store_true',
                    help='use dilated convolution')  # 使用扩张卷积
parser.add_argument('--precision', type=str, default='single',
                    choices=('single', 'half'),
                    help='FP precision for test (single | half)')

# Option for Residual dense network (RDN)  剩余密集网络（RDN）选项
parser.add_argument('--G0', type=int, default=64,
                    help='default number of filters. (Use in RDN)')
parser.add_argument('--RDNkSize', type=int, default=3,
                    help='default kernel size. (Use in RDN)')
parser.add_argument('--RDNconfig', type=str, default='B',
                    help='parameters config of RDN. (Use in RDN)')

# Option for Residual channel attention network (RCAN) # 剩余信道注意网络（RCAN）选项
parser.add_argument('--n_resgroups', type=int, default=10,
                    help='number of residual groups')
parser.add_argument('--reduction', type=int, default=16,
                    help='number of feature maps reduction')

# Training specifications  #训练 规格
parser.add_argument('--reset', action='store_true',
                    help='reset the training')  # 重新设置培训
parser.add_argument('--test_every', type=int, default=1000,
                    help='do test per every N batches')  # 每N批测试一次
parser.add_argument('--epochs', type=int, default=400,
                    help='number of epochs to train')  # 要培训的时段数
parser.add_argument('--batch_size', type=int, default=16,
                    help='input batch size for training')  # 输入培训批量
parser.add_argument('--split_batch', type=int, default=1,
                    help='split the batch into smaller chunks')  # 将批次分成小块
parser.add_argument('--self_ensemble', action='store_true',
                    help='use self-ensemble method for test')  # 用self-ensemble 方法进行测试
parser.add_argument('--test_only', action='store_true',
                    help='set this option to test the model')  # 设置此选项以测试模型
parser.add_argument('--demo_gen', action='store_true',
                    help='set this option to demo the model')  # 设置此选项以做demo
parser.add_argument('--gan_k', type=int, default=1,
                    help='k value for adversarial loss')

# Optimization specifications # 优化器规格
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')  # 学习速率
parser.add_argument('--decay', type=str, default='200',
                    help='learning rate decay type')  # 学习速率衰减类型 当epoch 等于200时 lr 除以 gamma
parser.add_argument('--gamma', type=float, default=0.5,
                    help='learning rate decay factor for step decay') # 阶跃衰减的学习速率衰减因子
parser.add_argument('--optimizer', default='ADAM',
                    choices=('SGD', 'ADAM', 'RMSprop'),
                    help='optimizer to use (SGD | ADAM | RMSprop)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--betas', type=tuple, default=(0.9, 0.999),
                    help='ADAM beta')
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight decay')
parser.add_argument('--gclip', type=float, default=0,
                    help='gradient clipping threshold (0 = no clipping)') # 梯度剪切阈值

# Loss specifications  # 损失规格
parser.add_argument('--loss', type=str, default='1*L1',
                    help='loss function configuration')
parser.add_argument('--skip_threshold', type=float, default='1e8',
                    help='skipping batch that has large error')

# Log specifications  # 日志规格
parser.add_argument('--save', type=str, default='',
                    help='file name to save')  # 要保存的文件名
parser.add_argument('--load', type=str, default='',
                    help='file name to load')  # 要加载的文件名
parser.add_argument('--resume', type=int, default=0,
                    help='resume from specific checkpoint')  # 从特定检查点恢复
parser.add_argument('--save_models', action='store_true',
                    help='save all intermediate models')  # 保存所有中间模型
parser.add_argument('--print_every', type=int, default=100,
                    help='how many batches to wait before logging training status')  # 记录培训状态前要等待多少批
parser.add_argument('--save_results', action='store_true',
                    help='save output results')  # 保存输出结果
parser.add_argument('--save_gt', action='store_true',
                    help='save low-resolution and high-resolution images together')  # 将低分辨率和高分辨率图像一起保存
# wdsr Option
parser.add_argument('--r_mean', type=float, default=0.4488,
                    help='Mean of R Channel')
parser.add_argument('--g_mean', type=float, default=0.4371,
                    help='Mean of G channel')
parser.add_argument('--b_mean', type=float, default=0.4040,
                    help='Mean of B channel')
parser.add_argument('--block_feats', type=int, default=512)

args = parser.parse_args()
template.set_template(args)

args.scale = list(map(lambda x: int(x), args.scale.split('+')))
args.data_train = args.data_train.split('+')
args.data_test = args.data_test.split('+')

if args.epochs == 0:
    args.epochs = 1e8

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False
