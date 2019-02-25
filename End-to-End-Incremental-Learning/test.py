import os

for root, dirs, files in os.walk('/home/tian/Desktop/image_resize/train'):
    if root == '/home/tian/Desktop/image_resize/train':
        print(dirs)