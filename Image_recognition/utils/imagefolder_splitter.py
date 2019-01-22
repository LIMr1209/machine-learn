from sklearn.model_selection import train_test_split
from utils.get_classes import get_classes


class ImageFolderSplitter:
    def __init__(self, path, train_size=0.8):
        self.path = path
        self.train_size = train_size
        self.x_train = []
        self.x_valid = []
        self.y_train = []
        self.y_valid = []
        self.data_x_path = get_classes(path)['data_x_path']
        self.data_y_label = get_classes(path)['data_y_label']
        # 80%的训练集，20%的测试机集
        self.x_train, self.x_valid, self.y_train, self.y_valid = train_test_split(self.data_x_path, self.data_y_label,
                                                                                  shuffle=True,
                                                                                  train_size=self.train_size)

    def getTrainingDataset(self):
        return self.x_train, self.y_train

    def getValidationDataset(self):
        return self.x_valid, self.y_valid


if __name__ == '__main__':
    splitter = ImageFolderSplitter(r"C:\Users\aaa10\Desktop\spiders\design\design\spiders\image")
