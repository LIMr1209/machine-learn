''' Incremental-Classifier Learning 
 Authors : Khurram Javed, Muhammad Talha Paracha
 Maintainer : Khurram Javed
 Lab : TUKL-SEECS R&D Lab
 Email : 14besekjaved@seecs.edu.pk '''
from PIL import Image
from torchvision import datasets, transforms
import torch
import numpy

from utils.imagefolder_splitter import ImageFolderSplitter


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
        instance = torch.from_numpy(numpy.random.uniform(low=-1, high=1, size=(32, 32))).float()
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

        self.train_data = datasets.CIFAR10("../data", train=True, transform=self.train_transform, download=False)

        self.test_data = datasets.CIFAR10("../data", train=False, transform=self.test_transform, download=False)


class Custom:
    def __init__(self, path='/home/tian/Desktop/image'):
        spitter = ImageFolderSplitter(path)
        self.classes = spitter.class_num
        self.name = 'custom'
        self.train_data = None
        self.test_data = None
        self.labels_per_class_train = len(spitter.x_train)
        self.labels_per_class_test = len(spitter.x_valid)

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

        self.train_data = trans(spitter.x_train, spitter.y_train, self.labels_per_class_train)

        self.test_data = trans(spitter.x_valid, spitter.y_valid, self.labels_per_class_test)


def trans(data, label, length):
    class A:
        def __init__(self):
            self.train_data = numpy.zeros(shape=(length,32,32,3))
            self.train_labels = label
            for i,j in enumerate(data):
                img = Image.open(j)
                img = img.convert("RGB")  # 如果有4通道图片转化为3通道
                img = numpy.array(img)
                self.train_data[i] = img

    return A()


if __name__ == '__main__':
    # MNIST()
    # CIFAR100()
    a = CIFAR10()
    c = Custom()
    d = a.train_data
    e = c.train_data
    b = 1

