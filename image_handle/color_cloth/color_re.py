from PIL import Image
import base64
from io import BytesIO
import requests
import numpy as np
from collections import Counter
from argparse import ArgumentParser


def get_colors(img_path, is_url=False):
    if is_url:
        response = requests.get(img_path)
        img = Image.open(BytesIO(response.content))
    else:
        img = Image.open(img_path)

    w, h = img.size
    if w > 50:
        new_width = 50
    else:
        new_width = w

    new_height = int((new_width * h) / w)
    img = img.resize((new_width, new_height))

    most_common_cloros = dict(Counter(list(img.getdata())).most_common())

    all_colors = list(most_common_cloros.keys())
    similar_colors = []
    main_colors = []

    while len(all_colors) != 0:
        diff = 90
        for color in all_colors[:1]:
            color_list = list(color)
            total_sum = sum(color_list)
            if color in similar_colors:
                continue

            for similar_color in all_colors:
                if similar_color == color or similar_color in similar_colors:
                    continue

                similar_color_list = list(similar_color)
                total_similar_color_sum = sum(similar_color_list)
                if len(similar_color_list) > 3 and len(color_list) > 3:
                    if similar_color_list[:3] == color_list[:3]:
                        diff = 190
                if abs(total_sum - total_similar_color_sum) <= diff:
                    similar_colors.append(similar_color)

            difference_color = list(set(all_colors).difference(set(similar_colors)))

            all_colors = difference_color

        main_colors.append(color)
        all_colors.remove(color)

    return main_colors


def get_args():
    Parser = ArgumentParser(
        description="Logorain color extractor is a simple image color extractor to get the main RGB colors in the image.")
    group = Parser.add_mutually_exclusive_group()
    group.add_argument('-i', '--imagePath', help="Define the image path in your local machine", type=str,
                       required=False)
    group.add_argument('-u', '--url', help="Define the image url", type=str, required=False)
    return Parser.parse_args()


if __name__ == '__main__':
    print(get_colors('2bb1fdb2cfb53965.jpg'))
    # args = get_args()
    # if args.imagePath:
    #     try:
    #         main_colors = get_colors(args.imagePath)
    #     except Exception as err:
    #         main_colors = None
    #         print("Error:", err)
    # elif args.url:
    #     try:
    #         main_colors = get_colors(args.url, is_url=True)
    #     except Exception as err:
    #         main_colors = None
    #         print("Error:", err)
    # else:
    #     raise Exception('You have to add an image path or url')
    #
    # print(main_colors)