import os
from collections import OrderedDict

# 获取目录下所有分类和图片数据
def get_classes(path):
    class2num = OrderedDict()
    data_x_path = []
    data_y_label = []
    for i, dirs in enumerate(os.listdir(path)):
        class2num[dirs] = i
        files = []
        for j, file in enumerate(os.listdir(os.path.join(path, dirs))):
            files.append(os.path.join(path, dirs, file))
            data_y_label.append(i)
        data_x_path.extend(sorted(files))
    return {'class2num': list(class2num.keys()), 'data_x_path': data_x_path, 'data_y_label': data_y_label}


if __name__ == '__main__':
    print(get_classes('/image/image')['class2num'])
