import os


# 获取目录下所有分类和图片数据
def get_classes(path):
    class2num = {}
    data_x_path = []
    data_y_label = []
    for root, dirs, files in os.walk(path):
        if len(files) == 0 and len(dirs) > 1:
            for i, dir1 in enumerate(dirs):
                class2num[dir1] = i
        elif len(files) > 1 and len(dirs) == 0:
            category = ""
            for key in class2num.keys():
                if key in root:
                    category = key
                    break
            label = class2num[category]
            files.sort()
            for i in range(len(files)):
                data_x_path.append(os.path.join(root, files[i]))
                data_y_label.append(label)
        else:
            raise RuntimeError("please check the folder structure!")
    return {'class2num': list(class2num.keys()), 'data_x_path': data_x_path, 'data_y_label': data_y_label}


if __name__ == '__main__':
    print(get_classes('/image/image')['class2num'])
