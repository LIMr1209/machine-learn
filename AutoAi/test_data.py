from autokeras.image.image_supervised import load_image_dataset
RESIZE_TRAIN_IMG_DIR = './data/resize/train'
RESIZE_TEST_IMG_DIR = './data/resize/test'

# Path to generate csv file
TRAIN_CSV_DIR = './train_labels.csv'
TEST_CSV_DIR = './test_labels.csv'

train_data, train_labels = load_image_dataset(csv_file_path=TRAIN_CSV_DIR, images_path=RESIZE_TRAIN_IMG_DIR)
test_data, test_labels = load_image_dataset(csv_file_path=TEST_CSV_DIR, images_path=RESIZE_TEST_IMG_DIR)
train_data = train_data.astype('float32') / 255
print('111'*100)

from keras.datasets import fashion_mnist
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# each image has to 3D: 2 coordinates, 1 value (gray scale)
X_train = X_train.reshape(X_train.shape + (1,))
X_test = X_test.reshape(X_test.shape + (1,))