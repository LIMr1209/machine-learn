from aip import AipImageClassify
import cv2
import math
import numpy as np
import requests
import time


def get_hue(a, b):
    if a == 0 or b == 0:
        h = 0
    else:
        h = math.atan2(b, a)
        h = (h / math.pi) * 180
    return h


def CIE2000_distance(lab1, lab2):
    # cie2000 色差公式

    lab1 = [lab1[0] / 255 * 100.0, lab1[1] - 128, lab1[2] - 128]
    lab2 = [lab2[0] / 255 * 100.0, lab2[1] - 128, lab2[2] - 128]

    c1 = math.sqrt(lab1[1] ** 2 + lab1[2] ** 2)
    c2 = math.sqrt(lab2[1] ** 2 + lab2[2] ** 2)

    c_mean = (c1 + c2) / 2.0

    G = 0.5 * (1 - math.sqrt(c_mean ** 7 / float(c_mean ** 7 + 25 ** 7)))

    a_1 = (1 + G) * lab1[1]
    a_2 = (1 + G) * lab2[1]

    C_prime_1 = math.sqrt(a_1 ** 2 + lab1[2] ** 2)
    C_prime_2 = math.sqrt(a_2 ** 2 + lab2[2] ** 2)

    h_1 = get_hue(a_1, lab1[2])
    h_2 = get_hue(a_2, lab2[2])

    delta_L = lab2[0] - lab1[0]
    delta_C = C_prime_2 - C_prime_1

    if C_prime_1 * C_prime_2 == 0:
        delta_h = 0
    else:
        if abs(h_2 - h_1) <= 180:
            delta_h = h_2 - h_1
        elif h_2 - h_1 > 180:
            delta_h = h_2 - h_1 - 360
        else:
            delta_h = h_2 - h_1 + 360

    delta_H = 2 * math.sqrt(c1 * c2) * math.sin(delta_h * math.pi / 2.0 * 180)

    l_mean = (lab1[0] + lab2[0]) / 2.0
    c_prime_mean = (C_prime_1 + C_prime_2) / 2.0

    if C_prime_1 * C_prime_2 == 0:
        h_mean = h_1 + h_2
    else:
        if abs(h_1 - h_2) <= 180:
            h_mean = (h_1 + h_2) / 2.0
        else:
            if h_1 + h_2 < 360:
                h_mean = (h_1 + h_2 + 360) / 2.0
            else:
                h_mean = (h_1 + h_2 - 360) / 2.0

    T = 1 - 0.17 * math.cos((h_mean - 30) * math.pi / 180.0) \
        + 0.24 * math.cos(2 * h_mean * math.pi / 180.0) \
        + 0.32 * math.cos((3 * h_mean + 6) * math.pi / 180.0) \
        - 0.2 * math.cos((4 * h_mean - 63) * math.pi / 180.0)

    delta_Phi = 30 * math.exp(-((h_mean - 275) / 25.0) ** 2)
    R_c = 2 * math.sqrt(c_prime_mean ** 7 / float(c_prime_mean ** 7 + 25 ** 7))
    S_l = 1 + (0.015 * (l_mean - 50) ** 2) / math.sqrt(20 + (l_mean - 50) ** 2)
    S_c = 1 + 0.045 * c_prime_mean
    S_h = 1 + 0.015 * c_prime_mean * T
    R_t = -math.sin(2 * delta_Phi * math.pi / 180.0) * R_c

    distance = math.sqrt((delta_L / S_l) ** 2
                         + (delta_C / S_c) ** 2
                         + (delta_H / S_h) ** 2
                         + R_t * (delta_C / S_c) * (delta_H / S_h))
    return distance


def LAB_shadow(LAB_color_1, LAB_color_2):
    # 当颜色较暗时，A和B中的值几乎保持不变。 但光的价值变化更大

    threshold_L = 70
    threshold_A = 15
    threshold_B = 20

    distance_in_L = math.fabs(LAB_color_1[0] - LAB_color_2[0])
    distance_in_A = math.fabs(LAB_color_1[1] - LAB_color_2[1])
    distance_in_B = math.fabs(LAB_color_1[2] - LAB_color_2[2])

    if distance_in_L < threshold_L \
            and distance_in_A < threshold_A \
            and distance_in_B < threshold_B:
        return True

    return False


def rgb2hex(rgb):
    rgb = rgb.split(',')
    color = "#"

    color += str(hex(int(rgb[0]))).replace('x', '0')[-2:]

    color += str(hex(int(rgb[1]))).replace('x', '0')[-2:]

    color += str(hex(int(rgb[2]))).replace('x', '0')[-2:]

    return color


APP_ID = '17887227'
API_KEY = 'HXkluVYkuRL1PURlsa973vsl'
SECRET_KEY = 'khKaBZEkN2EnDoeRkU18pGbBF71CH2Fd'


class ImageColor:
    def __init__(self, url=None, file_path=None):
        self.file_path = file_path
        self.client = AipImageClassify(APP_ID, API_KEY, SECRET_KEY)
        self.param = None
        self.url = url
        self.img = None
        self.response = None

    # 读取图片
    def read_image(self):
        if self.file_path:
            self.img = cv2.imread(self.file_path)
        elif self.url:
            self.response = requests.get(self.url)
            image_numpy = np.asarray(bytearray(self.response.content), dtype="uint8")
            img = cv2.imdecode(image_numpy, cv2.IMREAD_COLOR)
            self.img = img

    # 主体检测
    def subject_detection(self):
        if self.file_path:
            with open(self.file_path, 'rb') as fp:
                # 调用图像主体检测
                options = dict()
                options["with_face"] = 0
                response = self.client.objectDetect(fp.read(), options)
                self.param = response['result']
        elif self.url:
            # 调用图像主体检测
            options = dict()
            options["with_face"] = 0
            response = self.client.objectDetect(self.response.content, options)
            if 'result' in response:
                self.param = response['result']

    # 主体参数裁剪图片
    def tailoring(self):
        if self.param:
            self.img = self.img[self.param['top']:self.param['top'] + self.param['height'],
                            self.param['left']:self.param['left'] + self.param['width']]
        # cv2.imshow('ai', initial_image)  # 显示图片
        # cv2.waitKey(0)

    # 颜色识别
    def color_recognition(self, expected_size=40, in_clusters=7, out_clusters=3):
        # a = time.time()
        self.read_image()
        # b = time.time()
        # print('读取图片',b-a)
        # self.subject_detection()
        # c = time.time()
        # print('主体检测', c - b)
        # self.tailoring()
        # d = time.time()
        # print('裁剪', d - c)
        height, width = self.img.shape[:2]

        factor = math.sqrt(width * height / (expected_size * expected_size))

        # 下采样 缩小图片
        image = cv2.resize(self.img,
                           (int(width / factor), int(height / factor)),
                           interpolation=cv2.INTER_LINEAR)
        LAB_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
        frame_width = int(expected_size / 10 + 2)
        in_samples = []

        border_samples = []

        limit_Y = LAB_image.shape[0] - frame_width
        limit_X = LAB_image.shape[1] - frame_width

        for y in range(LAB_image.shape[0] - 1):
            for x in range(LAB_image.shape[1] - 1):
                pt = LAB_image[y][x]
                if x < frame_width or y < frame_width or y >= limit_Y or x >= limit_X:
                    border_samples.append(pt)
                else:
                    in_samples.append(pt)

        in_samples = np.array(in_samples, dtype=float)
        border_samples = np.array(border_samples, dtype=float)

        em_in = cv2.ml.EM_create()
        em_in.setClustersNumber(in_clusters)
        in_etval, in_likelihoods, in_labels, in_probs = em_in.trainEM(in_samples)

        in_means = em_in.getMeans()
        in_covs = em_in.getCovs()

        em_border = cv2.ml.EM_create()
        em_border.setClustersNumber(out_clusters)
        border_etval, border_likelihoods, border_labels, border_probs = em_border.trainEM(border_samples)

        border_means = em_border.getMeans()

        unique_border, counts_border = np.unique(border_labels, return_counts=True)
        count_border_labels = dict(zip(unique_border, counts_border))

        unique, counts = np.unique(in_labels, return_counts=True)

        count_in_labels = dict(zip(unique, counts))
        for i in range(in_clusters):
            if i not in count_in_labels:
                count_in_labels[i] = 0
        in_len = len(in_covs)

        valid = [True] * in_len

        # colors vs background
        for i in range(in_len):
            if not valid[i]:
                continue

            prop_in = float(count_in_labels[i]) / len(in_labels)

            # 如果比例太小，只能是按钮或标签
            if prop_in < 0.05:
                valid[i] = False
                continue

            # 删除相似的颜色
            for key in count_border_labels:
                prop_border = float(count_border_labels[key]) / len(border_labels)

                # 如果颜色更多地出现在中间，它就属于主色调，而不是背景。
                if prop_in > prop_border:
                    continue

                cie_dst = CIE2000_distance(in_means[i], border_means[key])

                if cie_dst < 5:
                    valid[i] = False

        # colors vs colors
        for i in range(in_len):
            if not valid[i]:
                continue

            for j in range(i + 1, in_len):
                if not valid[j]:
                    continue

                # 删除阴影和类似颜色
                cie_dst = CIE2000_distance(in_means[i], in_means[j])
                is_shadow = LAB_shadow(in_means[i], in_means[j])

                if is_shadow or cie_dst < 30:
                    if count_in_labels[j] > count_in_labels[i]:
                        valid[i] = False

                        count_in_labels[j] += count_in_labels[i]
                        break
                    else:
                        valid[j] = False
                        count_in_labels[i] += count_in_labels[j]

        num_valid = sum(True == x for x in valid)

        colors = []
        proportions = []
        total_color = 0

        # 如果布料与背景颜色相同，则采用更常见的颜色。
        if num_valid == 0:
            pos = max(count_in_labels, key=count_in_labels.get)
            colors = [in_means[pos]]
            proportions = [count_in_labels[pos]]
            total_color = count_in_labels[pos]

        for i in range(in_len):

            if not valid[i]:
                continue

            color = in_means[i]
            quantity = count_in_labels[i]
            colors.append(color)
            proportions.append(quantity)
            total_color += quantity

        color_list = []
        for i, color_LAB in enumerate(colors):
            color_LAB = np.array([[[color_LAB[0], color_LAB[1], color_LAB[2]]]])
            color_LAB = color_LAB.astype(np.uint8)
            color = cv2.cvtColor(color_LAB, cv2.COLOR_Lab2BGR)[0][0]
            color = color.tolist()

            color[0], color[2] = color[2], color[0]
            color = ','.join('%s' % id for id in color)
            color_weight = round(proportions[i] / sum(proportions), 2)
            color_list.append((color_weight, color, rgb2hex(color)))
        color_list = sorted(color_list, key=lambda d: d[0], reverse=True)
        e = time.time()
        # print('识别',e-d)
        return color_list


if __name__ == '__main__':
    image_color = ImageColor(url='https://p4.taihuoniao.com/image/201023/5f925531b8867a3622b5f161')
    print(image_color.color_recognition())
