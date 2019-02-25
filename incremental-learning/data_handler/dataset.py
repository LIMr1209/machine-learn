''' Incremental-Classifier Learning 
 Authors : Khurram Javed, Muhammad Talha Paracha
 Maintainer : Khurram Javed
 Lab : TUKL-SEECS R&D Lab
 Email : 14besekjaved@seecs.edu.pk '''
from PIL import Image
from torchvision import datasets, transforms
import torch
import numpy as np
import os


# To incdude a new Dataset, inherit from Dataset and add all the Dataset specific parameters here.
# Goal : Remove any data specific parameters from the rest of the code

class Dataset():
    '''
    Base class to reprenent a Dataset
    '''

    def __init__(self, classes, name, labels_per_class_train, labels_per_class_test):
        self.classes = classes
        self.name = name
        self.train_data = None
        self.test_data = None
        self.labels_per_class_train = labels_per_class_train
        self.labels_per_class_test = labels_per_class_test


class MNIST(Dataset):
    '''
    Class to include MNIST specific details
    '''

    def __init__(self):
        super().__init__(10, "MNIST", 6000, 1000)

        self.train_transform = transforms.Compose(
            [transforms.Resize((32, 32)),
             transforms.ToTensor(),
             transforms.Normalize((0.1307,), (0.3081,))])

        self.test_transform = transforms.Compose(
            [transforms.Resize((32, 32)), transforms.ToTensor(),
             transforms.Normalize((0.1307,), (0.3081,))])

        self.train_data = datasets.MNIST("data/mnist", train=True, transform=self.train_transform, download=False)

        self.test_data = datasets.MNIST("data/mnist", train=False, transform=self.test_transform, download=False)

    def get_random_instance(self):
        instance = torch.from_numpy(np.random.uniform(low=-1, high=1, size=(32, 32))).float()
        instance.unsqueeze_(0)
        return instance


class CIFAR100(Dataset):
    def __init__(self):
        super().__init__(100, "CIFAR100", 500, 100)

        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]

        self.train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        # DO NOT DO DATA NORMALIZATION; TO IMPLEMENT DATA NORMALIZATION, MAKE SURE THAT DATA NORMALIZATION IS STILL APPLIED IN GET_ITEM FUNCTION OF INCREMENTAL LOADER
        self.train_transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(),
             transforms.RandomCrop(32, padding=4),
             transforms.ToTensor(), ])

        self.test_transform = transforms.Compose(
            [transforms.ToTensor(), ])

        self.train_data = datasets.CIFAR100("data", train=True, transform=self.train_transform, download=False)

        self.test_data = datasets.CIFAR100("data", train=False, transform=self.test_transform, download=False)


class CIFAR10(Dataset):
    def __init__(self):
        super().__init__(10, "CIFAR10", 5000, 1000)

        self.train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

        self.train_transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(),
             transforms.RandomCrop(32, padding=4),
             transforms.ToTensor(), ])

        self.test_transform = transforms.Compose(
            [transforms.ToTensor(), ])

        self.train_data = datasets.CIFAR10("/home/tian/Desktop/data", train=True, transform=self.train_transform,
                                           download=False)

        self.test_data = datasets.CIFAR10("/home/tian/Desktop/data", train=False, transform=self.test_transform,
                                          download=False)


class Custom:
    def __init__(self, path='/home/tian/Desktop/image_resize/'):
        self.classes = 9
        self.name = 'custom'
        self.train_data = None
        self.test_data = None
        self.labels_per_class_train = 200
        self.labels_per_class_test = 100

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.test_transform = transforms.Compose([
            transforms.Resize(224),  # #缩放图片（Image）,保持长宽比不变，最短边为224像素
            transforms.CenterCrop(224),  # 在图片的中间区域进行裁剪
            transforms.ToTensor(),  # 转tensor
            normalize  # 归一化
        ])

        self.train_transform = transforms.Compose([
            transforms.Resize(256),  # #缩放图片（Image）,保持长宽比不变，最短边为224像素
            transforms.RandomResizedCrop(224),  # 在一个随机的位置进行裁剪
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])

        self.train_data, self.test_data = get_dataset(path)


class TrainDate:
    def __init__(self, train_data, train_labels):
        self.train_data = train_data
        self.train_labels = train_labels


class TestData:
    def __init__(self, test_data, test_labels):
        self.test_data = test_data
        self.test_labels = test_labels


def get_dataset(path):
    num_class = 9
    num_sample_train = 200
    num_sample_test = 100
    W = 224
    H = 224
    C = 3
    train_data = np.zeros((num_class * num_sample_train, W, H, C), dtype=np.uint8)
    test_data = np.zeros((num_class * num_sample_test, W, H, C), dtype=np.uint8)
    train_labels = []
    test_labels = []
    for i in range(num_class):
        train_labels.extend([i] * num_sample_train)
        test_labels.extend([i] * num_sample_test)
    train_i = 0
    test_i = 0
    for root, dirs, files in os.walk(path+'train'):
        if root == path+'train':
            label_list = dirs
            print(label_list)
        for file in files:
            img = np.array(Image.open(root + '/' + file))
            train_data[train_i] = img
            train_i += 1
    for root, dirs, files in os.walk(path+'test'):
        for file in files:
            img = np.array(Image.open(root + '/' + file))
            test_data[test_i] = img
            test_i += 1
    return TrainDate(train_data, train_labels), TestData(test_data, test_labels)


if __name__ == '__main__':
    c = Custom()
    numpy = c.train_data.train_data[200]  # 0-199 第一类  200-399 第二类
    print(c.train_data.train_labels[200])
    print(numpy.shape)
    img = Image.fromarray(numpy)
    img.show()
