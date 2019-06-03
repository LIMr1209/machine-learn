from PIL import Image
from torch.utils import data
from torchvision import transforms as T
from utils.imagefolder_splitter import ImageFolderSplitter
from config import opt


class DatasetFromFilename(data.Dataset):

    def __init__(self, root, transforms=None, flag=None):
        """
        主要目标： 获取所有图片的地址，并根据训练，验证，测试划分数据
        训练 train=True,Test=False
        验证 train=False,test=False
        测试 train=True,test=True
        """
        splitter = ImageFolderSplitter(root)
        self.flag = flag
        self.imgs, self.labels = splitter.getTrainingDataset()
        num = len(self.imgs)
        if self.flag == 'test':
            self.imgs, self.labels = splitter.getValidationDataset()  # 测试数据
        elif self.flag == 'train':
            self.imgs = self.imgs[int(0.2 * num):]
            self.labels = self.labels[int(0.2 * num):]
            # 训练数据
        elif self.flag == 'valid':
            self.imgs = self.imgs[:int(0.2 * num)]
            self.labels = self.labels[:int(0.2 * num)]

        if transforms is None:  # 转化器 图片转tensor
            # 将tensor正则化   mean 均值 std 方差 Normalized_image=(image-mean)/std
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            # 数据增强
            if self.flag == 'test':
                # 测试
                self.transforms = T.Compose([
                    T.Resize((opt.image_size, opt.image_size)),  # 缩放图片（Image）,保持长宽比不变，224x224
                    T.ToTensor(),  # 转tensor
                    normalize  # 归一化
                ])
            elif self.flag == 'train':
                # 训练
                self.transforms = T.Compose([
                    T.Resize(opt.image_size),  # #缩放图片（Image）,保持长宽比不变，最短边为224像素
                    T.CenterCrop(opt.image_size),  # 在图片的中间区域进行裁剪
                    T.ToTensor(),  # 转tensor
                    normalize  # 归一化
                ])
            elif self.flag == 'valid':
                # 验证
                # self.transforms = T.Compose([
                #     T.Resize((256, 256)),  # 缩放图片（Image）,保持长宽比不变，256x256
                #     T.RandomResizedCrop(opt.image_size),  # 在一个随机的位置进行裁剪,224x224
                #     T.RandomHorizontalFlip(p=0.5),  # 随机水平翻转给定的PIL.Image,概率为0.5
                #     T.RandomVerticalFlip(p=0.5),  # 随机垂直翻转给定的PIL.Image,概率为0.5
                #     T.RandomRotation(degrees=45),  # 随机翻转 (-45,45)度
                #     # T.ColorJitter(brightness=1, contrast=1, hue=0.5),  # 随机改变图像的亮度、对比度和饱和度
                #     T.ToTensor(),  # 转tensor
                #     normalize  # 归一化
                # ])
                self.transforms = T.Compose([
                    T.Resize(opt.image_size),  # #缩放图片（Image）,保持长宽比不变，最短边为224像素
                    T.CenterCrop(opt.image_size),  # 在图片的中间区域进行裁剪
                    T.ToTensor(),  # 转tensor
                    normalize  # 归一化
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
        data = data.convert("RGB")  # 如果有4通道图片转化为3通道
        data = self.transforms(data)
        return data, label, img_path ,opt.cate_classes[label]  # 返回数据级标签图片路径

    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    img = DatasetFromFilename(r'/image/image', flag='test')
    splitter = ImageFolderSplitter('/image/image')
    x_train, y_train = splitter.getTrainingDataset()
    x_valid, y_valid = splitter.getValidationDataset()
