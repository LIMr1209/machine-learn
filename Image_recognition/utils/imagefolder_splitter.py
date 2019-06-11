from sklearn.model_selection import train_test_split
from utils.get_classes import get_classes
from config import opt
import random


class ImageFolderSplitter:
    def __init__(self, path, train_size=0.8):
        self.path = path
        self.train_size = train_size
        self.x_train = []  # 训练图片
        self.x_test = []  # 测试图片
        self.y_train = []  # 训练标签
        self.y_test = []  # 测试标签
        self.tag_list = get_classes(path)['class2num']
        self.data_x_img = get_classes(path)['data_x_path']
        self.data_y_label = get_classes(path)['data_y_label']
        if opt.date_shuffle:
            # 随机80%的训练集和20%的测试集
            self.x_test, self.x_train, self.y_test, self.y_train = train_test_split(self.data_x_img,
                                                                                    self.data_y_label,
                                                                                    shuffle=True,
                                                                                    test_size=self.train_size)
        else:
            # 有序的80%的训练集和20%的测试集
            self.label = []
            self.img = []
            for i in range(1, self.data_y_label[-1] + 2):
                if i == 1:
                    self.label.append(self.data_y_label[:self.data_y_label.index(i)])
                    self.img.append(self.data_x_img[:self.data_y_label.index(i)])

                elif i == self.data_y_label[-1] + 1:
                    self.label.append(self.data_y_label[self.data_y_label.index(i - 1):])
                    self.img.append(self.data_x_img[self.data_y_label.index(i - 1):])
                else:
                    self.label.append(self.data_y_label[self.data_y_label.index(i - 1):self.data_y_label.index(i)])
                    self.img.append(self.data_x_img[self.data_y_label.index(i - 1):self.data_y_label.index(i)])
            for i in range(len(self.label)):
                x_test, x_train, y_test, y_train = train_test_split(self.img[i],
                                                                    self.label[i],
                                                                    shuffle=False,
                                                                    test_size=self.train_size)
                self.x_train.extend(x_train)
                self.x_test.extend(x_test)
                self.y_train.extend(y_train)
                self.y_test.extend(y_test)

            randnum = random.randint(0, 100)
            random.seed(randnum)
            random.shuffle(self.x_train)
            random.seed(randnum)
            random.shuffle(self.x_test)
            random.seed(randnum)
            random.shuffle(self.y_train)
            random.seed(randnum)
            random.shuffle(self.y_test)

    def getTrainingDataset(self):  # 返回训练集
        return self.x_train, self.y_train

    def getTestationDataset(self):  # 返回测试集
        return self.x_test, self.y_test


if __name__ == '__main__':
    splitter = ImageFolderSplitter('/home/tian/Desktop/spiders/design/design/spiders/image')
    x_train, y_train = splitter.getTrainingDataset()
    x_test, y_test = splitter.getTestationDataset()
