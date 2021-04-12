import os
import colorsys
from PIL import Image

# Hue 为 0 代表红色，120 代表绿色，240 代表蓝色。我们可以自定义 0-355 这 360 个数值，
def hueChange(img, hue):
    # It's better to raise an exception than silently return None if img is not
    # an Image.
    img.load()
    r, g, b = img.split()
    r_data = []
    g_data = []
    b_data = []
    for rd, gr, bl in zip(r.getdata(), g.getdata(), b.getdata()):
        h, s, v = colorsys.rgb_to_hsv(rd / 255., bl / 255., gr / 255.)
        rgb = colorsys.hsv_to_rgb(hue/360., s, v)
        rd, gr, bl = [int(x*255.) for x in rgb]
        r_data.append(rd)
        g_data.append(gr)
        b_data.append(bl)
    #
    r.putdata(r_data)
    g.putdata(g_data)
    b.putdata(b_data)
    # 红色：1､331
    # 黄色：31
    # 绿色：91､121､181
    # 蓝色：211､141
    # 紫色：271､301
    color = ''
    if hue in [1,331]:
        color = '红色'
    elif hue in [91,121,181]:
        color = '绿色'
    elif hue in [211,141]:
        color = '蓝色'
    elif hue in [271,301]:
        color = "紫色"
    elif hue == 31:
        color = "黄色"
    return Image.merge('RGB',(r,g,b)), color

if __name__ == '__main__':
    for i in os.listdir('origin_hsv'):
        basename, ext = os.path.splitext(i)
        img = Image.open('origin_hsv/{}'.format(i)).convert('RGB')
        for hue in range(1, 360, 30):
            img2,color = hueChange(img, hue)
            if not os.path.exists(basename):
                os.mkdir(basename)
            out_name = '{}/{}.jpg'.format(basename, hue)
            img2.save(out_name)
