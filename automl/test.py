import os, csv, cv2
from autokeras.image.image_supervised import load_image_dataset, ImageClassifier
from keras.models import load_model
from keras.utils import plot_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from imagefolder_splitter import ImageFolderSplitter
import imutils


# write csv
def write_csv(img_dir, csv_dir):
    list = []
    list.append(['File Name', 'Label'])
    for file_name in os.listdir(img_dir):
        for img in os.listdir("%s/%s" % (img_dir, file_name)):
            print(img)
            item = [file_name + "/" + img, file_name]
            list.append(item)
    f = open(csv_dir, 'w')
    writer = csv.writer(f)
    writer.writerows(list)


# resize images
def resize_img(RESIZE_TRAIN_IMG_DIR, RESIZE_TEST_IMG_DIR, splitter):
    for i, img_file in enumerate(splitter.x_train):
        cls_name = splitter.y_train[i]
        img = cv2.imread(img_file)
        img = cv2.resize(img, (RESIZE, RESIZE), interpolation=cv2.INTER_LINEAR)
        img_name = img_file[img_file.rfind('/') + 1:]
        if os.path.exists("%s/%s" % (RESIZE_TRAIN_IMG_DIR, cls_name)):
            cv2.imwrite("%s/%s/%s" % (RESIZE_TRAIN_IMG_DIR, cls_name, img_name), img)
        else:
            os.makedirs("%s/%s" % (RESIZE_TRAIN_IMG_DIR, cls_name))
            cv2.imwrite("%s/%s/%s" % (RESIZE_TRAIN_IMG_DIR, cls_name, img_name), img)
    for i, img_file in enumerate(splitter.x_test):
        cls_name = splitter.y_test[i]
        img = cv2.imread(img_file)
        img = cv2.resize(img, (RESIZE, RESIZE), interpolation=cv2.INTER_LINEAR)
        img_name = img_file[img_file.rfind('/') + 1:]
        if os.path.exists("%s/%s" % (RESIZE_TEST_IMG_DIR, cls_name)):
            cv2.imwrite("%s/%s/%s" % (RESIZE_TEST_IMG_DIR, cls_name, img_name), img)
        else:
            os.makedirs("%s/%s" % (RESIZE_TEST_IMG_DIR, cls_name))
            cv2.imwrite("%s/%s/%s" % (RESIZE_TEST_IMG_DIR, cls_name, img_name), img)


def train_autokeras(RESIZE_TRAIN_IMG_DIR, RESIZE_TEST_IMG_DIR, TRAIN_CSV_DIR, TEST_CSV_DIR, TIME):
    # Load images
    train_data, train_labels = load_image_dataset(csv_file_path=TRAIN_CSV_DIR, images_path=RESIZE_TRAIN_IMG_DIR)
    test_data, test_labels = load_image_dataset(csv_file_path=TEST_CSV_DIR, images_path=RESIZE_TEST_IMG_DIR)

    train_data = train_data.astype('float32') / 255
    test_data = test_data.astype('float32') / 255
    print("Train data shape:", train_data.shape)

    clf = ImageClassifier(verbose=True)
    clf.fit(train_data, train_labels, time_limit=TIME)
    clf.final_fit(train_data, train_labels, test_data, test_labels, retrain=True)

    y = clf.evaluate(test_data, test_labels)
    print("Evaluate:", y)
    # Save model architecture diagram
    # clf.load_searcher().load_best_model().produce_keras_model().save(MODEL_DIR)
    # clf.export_keras_model(MODEL_DIR)  # 储存

    model = load_model(MODEL_DIR)
    model.summary()  # 查看模型
    plot_model(model, to_file=MODEL_PNG)
    # Predict the category of the test image
    # img = cv2.imread(PREDICT_IMG_PATH)
    # img = cv2.resize(img, (RESIZE, RESIZE), interpolation=cv2.INTER_LINEAR)
    # x = img_to_array(img)
    # x = x.astype('float32') / 255
    # x = np.reshape(x, (1, RESIZE, RESIZE, 3))
    # print("x shape:", x.shape)
    #
    # y = clf.predict(x)
    # print("predict:", y)


def predict(MODEL_DIR, PREDICT_IMG_PATH, RESIZE):
    model = load_model(MODEL_DIR)

    # load the image
    image = cv2.imread(PREDICT_IMG_PATH)
    orig = image.copy()

    # pre-process the image for classification
    image = cv2.resize(image, (RESIZE, RESIZE))
    image = image.astype("float32") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    a = model.predict(image)
    # classify the input image
    result = model.predict(image)[0]
    # print (result.shape)
    proba = np.max(result)
    label = str(np.where(result == proba)[0])
    label = "{}: {:.2f}%".format(label, proba * 100)
    print(label)

    # draw the label on the image
    output = imutils.resize(orig, width=400)
    cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2)
    # show the output image
    cv2.imshow("Output", output)
    cv2.waitKey(0)


if __name__ == "__main__":
    # Folder for storing training images
    IMG_DIR = '/home/tian/Desktop/spiders/design/design/spiders/image'
    # Folder for storing testing images
    RESIZE_TRAIN_IMG_DIR = './data/resize/train'
    RESIZE_TEST_IMG_DIR = './data/resize/test'

    # Path to generate csv file
    TRAIN_CSV_DIR = './train_labels.csv'
    TEST_CSV_DIR = './test_labels.csv'

    # Path to test image
    PREDICT_IMG_PATH = '111.jpg'

    # Path to generate model file
    MODEL_DIR = 'Model.h5'
    MODEL_PNG = 'Model.png'

    # If your memory is not enough, please turn down this value.(my computer memory 16GB)
    RESIZE = 128
    # Set the training time, this is half an hour
    TIME = 0.5 * 60 * 60
    splitter = ImageFolderSplitter(IMG_DIR)
    print("Resize images...")
    # resize_img(RESIZE_TRAIN_IMG_DIR, RESIZE_TEST_IMG_DIR, splitter)
    print("write csv...")
    # write_csv(RESIZE_TRAIN_IMG_DIR, TRAIN_CSV_DIR)
    # write_csv(RESIZE_TEST_IMG_DIR, TEST_CSV_DIR)
    print("============Load...=================")
    train_autokeras(RESIZE_TRAIN_IMG_DIR, RESIZE_TEST_IMG_DIR, TRAIN_CSV_DIR, TEST_CSV_DIR, TIME)
    # predict(MODEL_DIR, PREDICT_IMG_PATH, RESIZE)
