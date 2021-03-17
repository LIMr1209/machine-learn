import colorsys


# rgb 转 cmyk
# k = min(255-r,255-g,255-b)
# c = 255-r-k
# m = 255-g-k
# y = 255-b-k
# rgb 转 hex
# RGB: 92, 184, 232
# 92 / 16 = 5
# 余12 -> 5C
# 184 / 16 = 11
# 余8 -> B8
# 232 / 16 = 14
# 余8 -> E8
# HEX = 5CB8E8


def get_dominant_color(image):
    # 颜色模式转换，以便输出rgb颜色值
    image = image.convert("RGBA")

    # 生成缩略图，减少计算量，减小cpu压力
    image.thumbnail((200, 200))

    max_score = 0
    dominant_color = 0

    for count, (r, g, b, a) in image.getcolors(image.size[0] * image.size[1]):

        # 跳过纯黑色
        if a == 0:
            continue

        saturation = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)[1]

        y = min(abs(r * 2104 + g * 4130 + b * 802 + 4096 + 131072) >> 13, 235)

        y = (y - 16.0) / (235 - 16)

        # 忽略高亮色
        if y > 0.9:
            continue

        # Calculate the score, preferring highly saturated colors.
        # Add 0.1 to the saturation so we don't completely ignore grayscale
        # colors by multiplying the count by zero, but still give them a low
        # weight.
        score = (saturation + 0.1) * count

        if score > max_score:
            max_score = score
            dominant_color = (r, g, b)

    return dominant_color


if __name__ == "__main__":
    from PIL import Image

    image = Image.open("image/0.jpg")
    print(get_dominant_color(image))
