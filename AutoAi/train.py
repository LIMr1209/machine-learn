import os, csv, cv2
from autokeras.image.image_supervised import load_image_dataset, ImageClassifier
from keras.models import load_model
from keras.utils import plot_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from imagefolder_splitter import ImageFolderSplitter
from autokeras.constant import Constant  # 微调参数设置  如batch_size  Image_size 等


# write csv
def write_csv(img_dir, csv_dir):
    list = []
    list.append(['File Name', 'Label'])
    for file_name in os.listdir(img_dir):
        for img in os.listdir("%s/%s" % (img_dir, file_name)):
            item = [file_name + "/" + img, file_name]
            list.append(item)
    f = open(csv_dir, 'w')
    writer = csv.writer(f)
    writer.writerows(list)


# resize images
def resize_img(path, data):
    for i, img_file in enumerate(data[0]):
        cls_name = data[1][i]
        img = cv2.imread(img_file)
        img = cv2.resize(img, (RESIZE, RESIZE), interpolation=cv2.INTER_LINEAR)
        img_name = img_file[img_file.rfind('/') + 1:]
        if os.path.exists("%s/%s" % (path, cls_name)):
            cv2.imwrite("%s/%s/%s" % (path, cls_name, img_name), img)
        else:
            os.makedirs("%s/%s" % (path, cls_name))
            cv2.imwrite("%s/%s/%s" % (path, cls_name, img_name), img)


def train_autokeras(RESIZE_TRAIN_IMG_DIR, RESIZE_TEST_IMG_DIR, TRAIN_CSV_DIR, TEST_CSV_DIR, TIME):
    # Load images
    train_data, train_labels = load_image_dataset(csv_file_path=TRAIN_CSV_DIR, images_path=RESIZE_TRAIN_IMG_DIR)  # 加载数据
    test_data, test_labels = load_image_dataset(csv_file_path=TEST_CSV_DIR, images_path=RESIZE_TEST_IMG_DIR)
    train_data = train_data.astype('float32') / 255
    test_data = test_data.astype('float32') / 255
    clf = ImageClassifier(verbose=True)
    clf.fit(train_data, train_labels, time_limit=TIME)  # 找最优模型
    clf.final_fit(train_data, train_labels, test_data, test_labels, retrain=True)  # 最优模型继续训练
    y = clf.evaluate(test_data, test_labels)
    print("测试集精确度:", y)
    score = clf.evaluate(train_data, train_labels)  # score: 0.8139240506329114
    print("训练集精确度:", score)
    clf.export_keras_model(MODEL_DIR)  # 储存


def predict(MODEL_DIR, PREDICT_IMG_PATH, RESIZE):
    model = load_model(MODEL_DIR)  # 加载模型
    # model.summary()  # 查看模型
    plot_model(model, to_file=MODEL_PNG)  # 保存模型结构为图片
    # load the image
    image = cv2.imread(PREDICT_IMG_PATH)  # 加载识别图片
    # pre-process the image for classification
    image = cv2.resize(image, (RESIZE, RESIZE))
    image = image.astype("float32") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    # classify the input image
    result = model.predict(image)[0]
    proba = np.max(result)
    a = np.where(result == proba)
    label = str(np.where(result == proba)[0])
    label = "{}: {:.2f}%".format(label, proba * 100)
    print(label)


if __name__ == "__main__":
    # Folder for storing training images
    IMG_DIR = '/home/tian/Desktop/spiders/design/design/spiders/image'
    # Folder for storing testing images
    RESIZE_TRAIN_IMG_DIR = './data/resize/train'
    RESIZE_TEST_IMG_DIR = './data/resize/test'

    # Path to generate csv file
    TRAIN_CSV_DIR = './data/train_labels.csv'
    TEST_CSV_DIR = './data/test_labels.csv'

    # Path to test image
    PREDICT_IMG_PATH = './data/水杯.jpg'

    # Path to generate model file
    MODEL_DIR = 'Model.h5'
    MODEL_PNG = 'Model.png'

    # If your memory is not enough, please turn down this value.(my computer memory 16GB)
    RESIZE = 128
    # Set the training time, this is half an hour
    TIME = 0.5 * 60 * 60  # 训练时间  半小时
    splitter = ImageFolderSplitter(IMG_DIR)
    # print("Resize images...")
    # resize_img(RESIZE_TRAIN_IMG_DIR, splitter.getTrainingDataset())
    # resize_img(RESIZE_TEST_IMG_DIR, splitter.getValidationDataset())
    # print("write csv...")
    # write_csv(RESIZE_TRAIN_IMG_DIR, TRAIN_CSV_DIR)
    # write_csv(RESIZE_TEST_IMG_DIR, TEST_CSV_DIR)
    print("============Load...=================")
    # train_autokeras(RESIZE_TRAIN_IMG_DIR, RESIZE_TEST_IMG_DIR, TRAIN_CSV_DIR, TEST_CSV_DIR, TIME)
    predict(MODEL_DIR, PREDICT_IMG_PATH, RESIZE)
