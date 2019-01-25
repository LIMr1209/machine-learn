from sklearn.model_selection import train_test_split
from utils.get_classes import get_classes


class ImageFolderSplitter:
    def __init__(self, path, train_size=0.8):
        self.path = path
        self.train_size = train_size
        self.x_train = []  # 训练图片
        self.x_valid = []  # 训练标签
        self.y_train = []  # 测试图片
        self.y_valid = []  # 测试标签
        self.data_x_path = get_classes(path)['data_x_path']
        self.data_y_label = get_classes(path)['data_y_label']
        # 随机80%的训练集和20%的测试集
        self.x_train, self.x_valid, self.y_train, self.y_valid = train_test_split(self.data_x_path, self.data_y_label,
                                                                                  shuffle=True,
                                                                                  train_size=self.train_size)

    def getTrainingDataset(self):  # 返回训练级
        return self.x_train, self.y_train

    def getValidationDataset(self):  # 返回测试集
        return self.x_valid, self.y_valid


if __name__ == '__main__':
    splitter = ImageFolderSplitter(r"C:\Users\aaa10\Desktop\spiders\design\design\spiders\image")
