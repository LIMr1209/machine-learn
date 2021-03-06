from PIL import Image
from torch.utils import data
from torchvision import transforms as T
from utils.imagefolder_splitter import ImageFolderSplitter


class DatasetFromFilename(data.Dataset):

    def __init__(self, root, transforms=None, test=False):
        """
        主要目标： 获取所有图片的地址，并根据训练，验证，测试划分数据
        训练 train=True,Test=False
        验证 train=False,test=False
        测试 train=True,test=True
        """
        splitter = ImageFolderSplitter(root)
        self.test = test
        if self.test:
            self.imgs, self.labels = splitter.getValidationDataset()  # 测试数据
        else:
            self.imgs, self.labels = splitter.getTrainingDataset()
        if transforms is None:  # 转化器 图片转tensor
            # 将tensor正则化   mean 均值 std 方差 Normalized_image=(image-mean)/std
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            # 数据增强
            if self.test:
                #  测试
                self.transforms = T.Compose([
                    T.Resize(224),  # #缩放图片（Image）,保持长宽比不变，最短边为224像素
                    T.CenterCrop(224),  # 在图片的中间区域进行裁剪
                    T.ToTensor(),  # 转tensor
                    normalize  # 归一化
                ])
            else:
                # 训练验证
                self.transforms = T.Compose([
                    T.Resize(256),  # #缩放图片（Image）,保持长宽比不变，最短边为224像素
                    T.RandomResizedCrop(224),  # 在一个随机的位置进行裁剪
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
        data = data.convert("RGB")  # 如果有4通道图片转化为3通道
        data = self.transforms(data)
        return data, label  # 返回数据级标签

    def __len__(self):
        return len(self.imgs)


def datasets_fn(root):
    test_data = DatasetFromFilename(root, test=True)  # 测试集
    train_data = DatasetFromFilename(root, test=False)  # 训练集
    return train_data, test_data


if __name__ == '__main__':
    img = DatasetFromFilename(r'/home/tian/Desktop/image', test=True)
    splitter = ImageFolderSplitter('/home/tian/Desktop/image')
    x_train, y_train = splitter.getTrainingDataset()
    x_valid, y_valid = splitter.getValidationDataset()
    a = 1
