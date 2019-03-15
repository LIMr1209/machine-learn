from sklearn.model_selection import train_test_split


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


import os


# 获取目录下所有分类和图片数据
def get_classes(path):
    class2num = {}
    num2class = {}
    class_nums = {}
    data_x_path = []
    data_y_label = []
    for root, dirs, files in os.walk(path):
        if len(files) == 0 and len(dirs) > 1:
            for i, dir1 in enumerate(dirs):
                num2class[i] = dir1
                class2num[dir1] = i
        elif len(files) > 1 and len(dirs) == 0:
            category = ""
            for key in class2num.keys():
                if key in root:
                    category = key
                    break
            label = class2num[category]
            class_nums[label] = 0
            for file1 in files:
                data_x_path.append(os.path.join(root, file1))
                data_y_label.append(label)
                class_nums[label] += 1
        else:
            raise RuntimeError("please check the folder structure!")
    return {'class2num': list(class2num.keys()), 'data_x_path': data_x_path, 'data_y_label': data_y_label}


if __name__ == '__main__':
    splitter = ImageFolderSplitter('/home/tian/Desktop/spiders/design/design/spiders/image')
    x_train, y_train = splitter.getTrainingDataset()
    x_valid, y_valid = splitter.getValidationDataset()
    print('11')
