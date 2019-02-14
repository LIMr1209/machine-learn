from utitls.imagefolder_splitter import ImageFolderSplitter
import os, csv, cv2

RESIZE = 128

IMG_DIR = '/home/tian/Desktop/spiders/design/design/spiders/image'
# Folder for storing testing images
RESIZE_TRAIN_IMG_DIR = './data/resize/train'
RESIZE_TEST_IMG_DIR = './data/resize/test'

# Path to generate csv file
TRAIN_CSV_DIR = './data/train_labels.csv'
TEST_CSV_DIR = './data/test_labels.csv'


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
        try:
            img = cv2.imread(img_file)
            img = cv2.resize(img, (RESIZE, RESIZE), interpolation=cv2.INTER_LINEAR)
        except:
            print(img_file)
            continue
        img_name = img_file[img_file.rfind('/') + 1:]
        if os.path.exists("%s/%s" % (path, cls_name)):
            cv2.imwrite("%s/%s/%s" % (path, cls_name, img_name), img)
        else:
            os.makedirs("%s/%s" % (path, cls_name))
            cv2.imwrite("%s/%s/%s" % (path, cls_name, img_name), img)


if __name__ == '__main__':
    splitter = ImageFolderSplitter(IMG_DIR)
    # print("Resize images...")
    resize_img(RESIZE_TRAIN_IMG_DIR, splitter.getTrainingDataset())
    resize_img(RESIZE_TEST_IMG_DIR, splitter.getValidationDataset())
    # print("write csv...")
    write_csv(RESIZE_TRAIN_IMG_DIR, TRAIN_CSV_DIR)
    write_csv(RESIZE_TEST_IMG_DIR, TEST_CSV_DIR)
