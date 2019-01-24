# coding:utf8
from PIL import Image
from torch.utils import data
from torchvision import transforms as T
from utils.imagefolder_splitter import ImageFolderSplitter
from config import opt
import cv2


class DatasetFromFilename(data.Dataset):

    def __init__(self, root, transforms=None, train=True, test=False):
        """
        主要目标： 获取所有图片的地址，并根据训练，验证，测试划分数据
        训练 train=True,Test=False
        验证 train=False,test=False
        测试 train=True,test=True
        """
        splitter = ImageFolderSplitter(root)
        self.train = train
        self.test = test
        if self.test:
            self.imgs, self.labels = splitter.getValidationDataset()  # 测试数据
        else:
            self.imgs, self.labels = splitter.getTrainingDataset()
            num = len(self.imgs)
            if self.train:  # 训练数据
                self.imgs = self.imgs[:int(0.7 * num)]
                self.labels = self.labels[:int(0.7 * num)]
            else:  # 验证数据
                self.imgs = self.imgs[int(0.7 * num):]
                self.labels = self.labels[:int(0.7 * num)]

        if transforms is None:  # 转化器 图片转tensor
            # 将tensor正则化   mean 均值 std 方差 Normalized_image=(image-mean)/std
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            # 数据增强
            if self.test or not self.train:
                # 训练 测试
                self.transforms = T.Compose([
                    T.Resize(opt.image_size),  # #缩放图片（Image）,保持长宽比不变，最短边为224像素
                    T.CenterCrop(opt.image_size),  # 在图片的中间区域进行裁剪
                    T.ToTensor(),  # 转tensor
                    normalize  # 归一化
                ])
            else:
                # 验证
                self.transforms = T.Compose([
                    T.Resize(256),  # #缩放图片（Image）,保持长宽比不变，最短边为224像素
                    T.RandomResizedCrop(opt.image_size),  # 在一个随机的位置进行裁剪
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    normalize
                ])

    def __getitem__(self, index):
        """
        一次返回一张图片的数据
        """
        img_path = self.imgs[index]
        label = self.labels[index]
        # img = cv2.imread(img_path)
        # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        data = Image.open(img_path)
        data = data.convert("RGB")
        data = self.transforms(data)
        return data, label

    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    img = DatasetFromFilename(r'/home/tian/Desktop/spiders/design/design/spiders/image_test')
    for i, (img, label) in enumerate(img):
        print(img.size())
        print(label)
