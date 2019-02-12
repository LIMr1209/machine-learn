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
            label = category
            class_nums[label] = 0
            for file1 in files:
                data_x_path.append(os.path.join(root, file1))
                data_y_label.append(label)
                class_nums[label] += 1
        else:
            raise RuntimeError("please check the folder structure!")
    return {'class2num': list(class2num.keys()), 'data_x_path': data_x_path, 'data_y_label': data_y_label}
