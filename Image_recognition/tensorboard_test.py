import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import numpy as np
from datetime import datetime

# Writer will output to ./runs/ directory by default
train_writer = SummaryWriter(log_dir='./runs/train_' + datetime.now().strftime('%y%m%d-%H-%M-%S'))
value_writer = SummaryWriter(log_dir='./runs/val_' + datetime.now().strftime('%y%m%d-%H-%M-%S'))
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = datasets.MNIST('mnist', train=True, download=True, transform=transform)
valset = datasets.MNIST('mnist', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
valloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
model = torchvision.models.resnet50(False).train()
# Have ResNet model take in grayscale rather than RGB
model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
for j in range(10):
    for i, (images, labels) in enumerate(trainloader):
        grid = torchvision.utils.make_grid(images)
        train_writer.add_image('train_images', grid, i)  # 图片
        # train_writer.add_graph(model, images)  # 网络结构
        train_writer.add_scalar('train_loss', i, i)  # 标量  误差走势图
        # writer.add_image_with_boxes('imagebox', grid, torch.Tensor([[10, 10, 40, 40], [40, 40, 60, 60]]), i)
        train_writer.add_text('train_ext', 'text logged at step:' + str(i), i)  # 准确率文本
        a = torch.from_numpy(np.random.randint(2, size=100))
        # train_writer.add_scalars('data/scalar_group', {'x': i,
        #                                          'y': i,
        #                                          'loss': i}, i)
        b = torch.from_numpy(np.random.rand(100))
        train_writer.add_pr_curve('train_acc', a, b, i)  # 精确率
    train_writer.close()

    for i, (images, labels) in enumerate(valloader):
        grid = torchvision.utils.make_grid(images)
        value_writer.add_image('val_images', grid, i)  # 图片
        # value_writer.add_graph(model, images)  # 网络结构
        value_writer.add_scalar('val_loss', i, i)  # 标量  误差走势图
        # writer.add_image_with_boxes('imagebox', grid, torch.Tensor([[10, 10, 40, 40], [40, 40, 60, 60]]), i)
        value_writer.add_text('val_text', 'text logged at step:' + str(i), i)  # 准确率文本
        a = torch.from_numpy(np.random.randint(2, size=100))
        b = torch.from_numpy(np.random.rand(100))
        train_writer.add_pr_curve('train_acc', a, b, i)  # 精确率
    value_writer.close()

# matplotlib
# import matplotlib.pyplot as plt
# plt.switch_backend('agg')
#
# fig = plt.figure()
#
# c1 = plt.Circle((0.2, 0.5), 0.2, color='r')
# c2 = plt.Circle((0.8, 0.5), 0.2, color='r')
#
# ax = plt.gca()
# ax.add_patch(c1)
# ax.add_patch(c2)
# plt.axis('scaled')
#
#
# writer = SummaryWriter()
# writer.add_figure('matplotlib', fig)
# writer.close()


# import time
# try:
#     import nvidia_smi
#     nvidia_smi.nvmlInit()
#     handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)  # gpu0
# except ImportError:
#     print('This demo needs nvidia-ml-py or nvidia-ml-py3')
#     exit()
#
#
# with SummaryWriter() as writer:
#     for n_iter in range(50):
#         res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
#         writer.add_scalar('nv/gpu', res.gpu, n_iter)
#         res = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
#         writer.add_scalar('nv/gpu_mem', res.used, n_iter)
#         time.sleep(0.1)
