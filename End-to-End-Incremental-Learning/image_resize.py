from PIL import Image
import os.path


def convertjpg(width=224, height=224):
    for root, dirs, files in os.walk('/home/tian/Desktop/image_增量/'):
        for file in files:
            img = Image.open(root + '/' + file)
            img = img.convert('RGB')
            try:
                root_resize = root.replace('image_增量', 'image_resize')
                if not os.path.exists(root_resize):
                    os.makedirs(root_resize)
                new_img = img.resize((width, height), Image.ANTIALIAS)
                if not os.path.exists(root_resize + '/' + file):
                    new_img.save(root_resize + '/' + file)
            except Exception as e:
                print(e)


convertjpg()
