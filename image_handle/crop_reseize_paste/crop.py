import numpy as np
from PIL import Image

# crop 草图 规范
def crop_and_normalize(pil_img, offset=20):
    w, h = pil_img.size
    img_array = np.array(pil_img.convert('L'))
    boundary = np.argwhere(img_array == 0)
    x = [element[1] for element in boundary]
    y = [element[0] for element in boundary]

    if min(x) - offset < 0:
        left = min(x)
    else:
        left = min(x) - offset
    if max(x) + offset > w:
        right = max(x)
    else:
        right = max(x) + offset
    if min(y) - offset < 0:
        up = min(y)
    else:
        up = min(y)- offset
    if max(y) + offset > h:
        down = max(y)
    else:
        down = max(y) + offset

    # long_edge = max(max(x) - min(x), max(y) - min(y))
    # half = int(long_edge / 2)
    # centerX, centerY = int((min(x) + max(x)) / 2), int((min(y) + max(y)) / 2)
    #
    # left, up, right, down = centerX - half - offset, centerY - half - offset, centerX + half + offset, centerY + half + offset
    #
    # left = 0 if left < 0 else left
    # up = 0 if up < 0 else up
    # right = 256 if right > 256 else right
    # down = 256 if down > 256 else down

    box = (left, up, right, down)
    crop_img = pil_img.crop(box)

    return crop_img


def crop_and_normalize_1(pil_img, offset=20):
    w,h = pil_img.size
    for x in range(0, w):
        for y in range(0, h):
            a, b, c, d = pil_img.getpixel((x, y))
            if a < 10 or b<10 or c <10:
                if x - offset < 0:
                    left = x
                else:
                    left = x - offset
                break
        else:
            continue
        break
    for y in range(0, h):
        for x in range(0, w):
            a, b, c, d = pil_img.getpixel((x, y))
            if a < 10 or b<10 or c <10:
                if y - offset < 0:
                    up = y
                else:
                    up = y - offset
                break
        else:
            continue
        break
    for x in range(w-1, 0,-1):
        for y in range(h-1,0,-1):
            a, b, c, d = pil_img.getpixel((x, y))
            if a < 10 or b<10 or c <10:
                if x + offset > w:
                    right = x
                else:
                    right = x + offset
                break
        else:
            continue
        break
    for y in range(h - 1, 0, -1):
        for x in range(w - 1, 0, -1):
            a, b, c,d  = pil_img.getpixel((x, y))
            if a < 10 or b<10 or c <10:
                if y + offset > h:
                    down = y
                else:
                    down = y + offset
                break
        else:
            continue
        break

    box = (left, up, right, down)
    crop_img = pil_img.crop(box)

    return crop_img

import time
a = time.time()
img = Image.open('下载.png')
new_img = crop_and_normalize_1(img)
new_img.save('下载_1new.png')
print(time.time()-a)
