import pickle
import numpy as np
import os
from PIL import Image


def load_data(dataset):
    """
    Parameters
    ----------
    dataset: the name of a dataset

    Returns
    ----------
    images_train: a tensor of C * N * C * W * H
    labels_train: a tensor of C * N * 1
    images_val: a tensor of C * N * C * W * H
    labels_val: a tensor of C * N * 1
    images_test: a tensor of C * N * C * W * H
    labels_test: a tensor of C * N * 1
    """

    if dataset == 'cifar100':
        num_class = 100
        num_sample_train = 500
        num_sample_test = 100
        num_sample_val = 0
        W = 32
        H = 32
        C = 3

        # load data

        with open('../../data/cifar-100-python/train', 'rb') as f:
            data = pickle.load(f, encoding='bytes')
            fine_labels_train = data[b'fine_labels']
            fine_labels_train = np.asarray(fine_labels_train)
            coarse_labels_train = data[b'coarse_labels']
            coarse_labels_train = np.asarray(coarse_labels_train)
            data_train = data[b'data']
            data_train = data_train.reshape((-1, 3, 32, 32))
            # data_train = np.asarray(data_train, dtype=np.float32)
            # data_train = data_train.transpose((0, 2, 3, 1))

        # plt.imshow(data_train[3])
        # plt.show()

        with open('../../data/cifar-100-python/test', 'rb') as f:
            data = pickle.load(f, encoding='bytes')
            fine_labels_test = data[b'fine_labels']
            fine_labels_test = np.asarray(fine_labels_test)
            coarse_labels_test = data[b'coarse_labels']
            coarse_labels_test = np.asarray(coarse_labels_test)
            data_test = data[b'data']
            data_test = data_test.reshape((-1, 3, 32, 32))
            # data_test = data_test.transpose((0, 2, 3, 1))

        images_train = np.zeros((num_class, num_sample_train, C, W, H), dtype=np.uint8)
        labels_train = np.zeros((num_class, num_sample_train), dtype=int)
        images_val = 0
        labels_val = 0
        images_test = np.zeros((num_class, num_sample_test, C, W, H), dtype=np.uint8)
        labels_test = np.zeros((num_class, num_sample_test), dtype=int)

        for i in range(num_class):
            idx = fine_labels_train == i
            images_train[i] = data_train[idx]
            labels_train[i] = fine_labels_train[idx]

            idx = fine_labels_test == i
            images_test[i] = data_test[idx]
            labels_test[i] = fine_labels_test[idx]

        return images_train, labels_train, images_val, labels_val, images_test, labels_test

    elif dataset == 'custom':
        images_val = 0
        labels_val = 0
        num_class = 9
        num_sample_train = 200
        num_sample_test = 100
        num_sample_val = 0
        W = 224
        H = 224
        C = 3
        images_train = np.zeros((num_class, num_sample_train, C, W, H), dtype=np.uint8)
        labels_train = np.zeros((num_class, num_sample_train), dtype=int)
        images_test = np.zeros((num_class, num_sample_test, C, W, H), dtype=np.uint8)
        labels_test = np.zeros((num_class, num_sample_test), dtype=int)
        image_train = np.zeros((num_sample_train * num_class, C, W, H), dtype=np.uint8)
        image_test = np.zeros((num_sample_test * num_class, C, W, H), dtype=np.uint8)
        fine_labels_train = []
        fine_labels_test = []
        for i in range(num_class):
            fine_labels_train.extend([i] * num_sample_train)
            fine_labels_test.extend([i] * num_sample_test)
        fine_labels_train = np.asarray(fine_labels_train)
        fine_labels_test = np.asarray(fine_labels_test)
        train_i = 0
        test_i = 0
        for root, dirs, files in os.walk('/home/tian/Desktop/image_resize/train'):
            if root == '/home/tian/Desktop/image_resize/train':
                label_list = dirs
                print(label_list)
            for file in files:
                img = np.array(Image.open(root + '/' + file))
                try:
                    img = img.transpose(2, 0, 1)
                except:
                    print(img.shape)
                    print(root + '/' + file)
                image_train[train_i] = img
                train_i += 1
        for root, dirs, files in os.walk('/home/tian/Desktop/image_resize/test'):
            for file in files:
                img = np.array(Image.open(root + '/' + file))
                try:
                    img = img.transpose(2, 0, 1)
                except:
                    print(img.shape)
                    print(root + '/' + file)
                image_test[test_i] = img
                test_i += 1
        for i in range(num_class):
            idx = fine_labels_train == i
            images_train[i] = image_train[idx]
            labels_train[i] = fine_labels_train[idx]

            idx = fine_labels_test == i
            images_test[i] = image_test[idx]
            labels_test[i] = fine_labels_test[idx]
        return images_train, labels_train, images_val, labels_val, images_test, labels_test
    else:
        print('Error: The dataset name is unknown.')


if __name__ == '__main__':
    b = load_data('custom')
    numpy = b[0][8][199]
    numpy = numpy.transpose(1, 2, 0)
    img = Image.fromarray(numpy)
    img.show()
