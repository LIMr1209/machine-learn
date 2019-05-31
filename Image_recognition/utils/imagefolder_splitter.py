from sklearn.model_selection import train_test_split
from utils.get_classes import get_classes
from config import opt


class ImageFolderSplitter:
    def __init__(self, path, train_size=0.8):
        self.path = path
        self.train_size = train_size
        self.x_train = []  # 训练图片
        self.x_valid = []  # 测试图片
        self.y_train = []  # 训练标签
        self.y_valid = []  # 测试标签
        self.data_x_img = get_classes(path)['data_x_path']
        self.data_y_label = get_classes(path)['data_y_label']
        if opt.date_shuffle:
            # 随机80%的训练集和20%的测试集
            self.x_train, self.x_valid, self.y_train, self.y_valid = train_test_split(self.data_x_img,
                                                                                      self.data_y_label,
                                                                                      shuffle=True,
                                                                                      train_size=self.train_size)
        else:
            # 有序的80%的训练集和20%的测试集
            self.x_train = self.data_x_img[int(len(self.data_x_img) * 0.2):]
            self.x_valid = self.data_x_img[:int(len(self.data_x_img) * 0.2)]
            self.y_train = self.data_y_label[int(len(self.data_x_img) * 0.2):]
            self.y_valid = self.data_y_label[:int(len(self.data_x_img) * 0.2)]

    def getTrainingDataset(self):  # 返回训练集
        return self.x_train, self.y_train

    def getValidationDataset(self):  # 返回测试集
        return self.x_valid, self.y_valid


if __name__ == '__main__':
    splitter = ImageFolderSplitter('/home/tian/Desktop/spiders/design/design/spiders/image')
    x_train, y_train = splitter.getTrainingDataset()
    x_valid, y_valid = splitter.getValidationDataset()
    print('11')
