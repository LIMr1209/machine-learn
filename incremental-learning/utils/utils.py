import numpy as np
import time
import torch


def resize_image(img, factor):
    '''
    
    :param img: 
    :param factor: 
    :return: 
    '''
    img2 = np.zeros(np.array(img.shape) * factor)

    for a in range(0, img.shape[0]):
        for b in range(0, img.shape[1]):
            img2[a * factor:(a + 1) * factor, b * factor:(b + 1) * factor] = img[a, b]
    return img2


# 保存模型参数以及优化器参数
def save_checkpoint(state):
    prefix = './checkpoint/ResNet152_'
    filename = time.strftime(prefix + '%m%d_%H-%M-%S.pth.tar')
    torch.save(state, filename)
