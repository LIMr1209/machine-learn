import os
from PIL import Image, ImageDraw
import requests
import re
import numpy as np
import scipy.cluster


#  读取图片并返回ImageExtractor 实例化对象
def read_file(image_path):
    if not os.path.exists(image_path):
        raise ValueError('Image path {} not exist.'.format(image_path))
    img = ImageExtractor()
    img.raw_image = Image.open(image_path)
    img.initialized = True
    return img


class ImageExtractor(object):
    def __init__(self):
        # 加工图
        self.image = None
        # 原图
        self.raw_image = None
        self.image_array = None
        # 颜色
        self.tones = None  # rgb
        self.hex = None  # hex
        self.tones_str = []
        self.initialized = False
        self.tone_image = None

    # 获取加工图
    def reduce_size(self, max_width):
        # 原图尺寸
        h, w = self.raw_image.size[0], self.raw_image.size[1]
        if w <= max_width:
            self.image = self.raw_image
            return self
        new_w = int(max_width)
        new_h = int(new_w / w * h)
        # 加工图
        self.image = self.raw_image.resize((new_h, new_w), Image.ANTIALIAS)

    # 像素
    def unstack_pixel(self):
        image_array = np.asarray(self.image)
        shape = image_array.shape
        # This will `unstack` the original array into a linear fashion.
        # scipy.product(shape[:2]) get the total number of points in the images
        # reshape(x, 3) reshape the current 3d-array from h*w*3 => (h*w)*3
        self.image_array = image_array.reshape(scipy.product(shape[:2]), shape[2]).astype(float)

    # 获取颜色 获取num中
    def extract_tones(self, num):
        image_array = self.image_array
        codes, dist = scipy.cluster.vq.kmeans(image_array, num)  # k-means 聚类方法
        # vecs, dist = scipy.cluster.vq.vq(image_array, codes)
        # counts, bins = scipy.histogram(vecs, len(codes))
        codes = [[int(_) for _ in code] for code in codes]
        self.tones = codes
        for i in self.tones:
            j = ",".join('%s' % z for z in i)
            self.tones_str.append(j)

    # 打印顔色
    def print_tones(self):
        for ind, each_tone in enumerate(self.tones, 1):
            print(f'Tone {ind}: {each_tone}')

    # 結合顔色
    def combine_tones(self):
        image_width, image_height = self.image.size[0], self.image.size[1]

        width = round(image_width / len(self.tones))
        height = width

        tone_image = Image.new('RGB', (image_width, height + image_height))
        tone_image.paste(self.image, (0, 0))

        for ind, each_tone in enumerate(self.tones):
            tone_tuple = (each_tone[0], each_tone[1], each_tone[2])
            temp_image = Image.new('RGB', (width, height), tone_tuple)
            tone_image.paste(temp_image, (width * ind, image_height))
        self.tone_image = tone_image

    # 添加色边框
    def add_borders(self):
        image_width, image_height = self.image.size[0], self.image.size[1]
        width = round(image_width / len(self.tones))
        height = width

        border_width = 10
        border_color = self.__get_board_color()
        draw = ImageDraw.Draw(self.tone_image)
        draw.line(xy=((0, image_height), (image_width - 1, image_height)),
                  fill=border_color,
                  width=border_width)
        for ind, _ in enumerate(self.tones[:-1], 1):
            draw.line(xy=((width * ind, image_height), (width * ind, image_height + height)),
                      fill=border_color,
                      width=int(width / 20))

    def __get_board_color(self):
        return 'white'

    def rgb2hex(self):
        self.hex = []
        for rgb in self.tones:
            strs = "#"
            for j in range(0, 3):
                s = hex(rgb[j])[2:]
                if len(s) < 2:
                    s += '0'
                strs += s
            self.hex.append(strs)

    def rgb_to_cmyk(self):
        data = {'RGB_R': '203', 'RGB_G': '72', 'RGB_B': '145', 'rgb_iccprofile': '0', 'cmyk_iccprofile': '9',
                'intent': '3'}
        host = 'https://www.colortell.com/rgb2cmyk'
        response = requests.post(host, data=data)
        rex = re.compile(r'<td><code>(\d*?).</code></td>', re.M)
        result = rex.findall(response.text)

    def rgb_to_pantone(self):
        pass


def main():
    img = read_file('image/0.jpg')
    img.reduce_size(img.raw_image.size[1])
    img.unstack_pixel()
    img.extract_tones(4)
    img.rgb2hex()
    print(str(img.tones))
    print(img.hex)


if __name__ == '__main__':
    main()
