import os
from collections import OrderedDict


# 获取目录下所有分类和图片数据
def get_classes(path):
    class2num = OrderedDict()
    data_x_path = []
    data_y_label = []
    for i, dirs in enumerate(sorted(os.listdir(path))):
        class2num[dirs] = i
        files = []
        for j, file in enumerate(os.listdir(os.path.join(path, dirs))):
            files.append(os.path.join(path, dirs, file))
            data_y_label.append(i)
        data_x_path.extend(sorted(files))
    return {'class2num': list(class2num.keys()), 'data_x_path': data_x_path, 'data_y_label': data_y_label,
            'classes_dict': class2num}


def find_classes(path):
    classes = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    images = []
    dir = os.path.expanduser(path)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                item = (path, class_to_idx[target])
                images.append(item)

    return images, classes, class_to_idx


def get_new_train(old_path, new_path):
    new_img = []
    new_label = []
    classes_dict = get_classes(old_path)
    for dirs in os.listdir(new_path):
        label = classes_dict[dirs]
        for file in os.listdir(os.path.join(new_path, dirs)):
            new_img.append(file)
            new_label.append(label)
    return new_img, new_label


if __name__ == '__main__':
    a = get_classes('/image/image')
    b = find_classes('/image/image')
