import cv2
import os
import numpy as np
import shutil

from PIL import Image

img_dir = "test"
img_dir = os.path.join(os.getcwd(), img_dir)
result_dir = os.path.join(os.getcwd(),"Results")


# get Region of Interest
def getROI(mask: np.array) -> list:
    x_left = 0
    y_top = 0
    x_right = mask.shape[0]
    y_bottom = mask.shape[1]

    ## region of interest through cols
    cols = np.sum(mask, axis = 0)
    for i in range(0, cols.shape[0]):
        if cols[i] != 0:
            y_top = i
            break
    for i in range(0, cols.shape[0]):
        if cols[-i] != 0:
            y_bottom= cols.shape[0] - i
            break

    ## region of interest through rows
    rows = np.sum(mask, axis = 1)
    for i in range(0, rows.shape[0]):
        if rows[i] != 0:
            x_left = i
            break
    for i in range(0, rows.shape[0]):
        if rows[-i] != 0:
            x_right = rows.shape[0] - i
            break

    return [x_left, x_right, y_top, y_bottom]



# output the major color of the suitcase
# except "white"
def majoColor_inrange(image_path)-> str:
    color_range = {
        "黑色": [np.array([0,0,0]),np.array([180,255,80])],
        "灰色":[np.array([0,0,80]),np.array([180,15,220])],
        "白色":[np.array([0,0,221]),np.array([180,30,255])],
        "红色":[np.array([0,15,46]),np.array([10,255,255])],
        "红色2":[np.array([156,15,46]),np.array([180,255,255])],
        "橙色":[np.array([11,15,46]),np.array([25,255,255])],
        "黄色":[np.array([26,15,46]),np.array([34,255,255])],
        "绿色":[np.array([35,15,46]),np.array([77,255,255])],
        "青色":[np.array([78,15,46]),np.array([99,255,255])],
        "蓝色":[np.array([100,15,46]),np.array([124,255,255])],
        "紫色":[np.array([125,15,46]),np.array([155,255,255])]
    }
    color = "其他"
    image = Image.open(image_path)
    # image.show()
    img = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    # img = cv2.imread(os.path.join(image_path))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    other_mask = cv2.inRange(img, np.float32(color_range["白色"][0]), np.float32(color_range["白色"][1]))
    white_mask = cv2.bitwise_not(other_mask)
    [x1, x2, y1, y2] = getROI(white_mask)
    img = img[x1:x2, y1:y2]
    img = cv2.blur(img, (7,7))
    
    color_list = []

    for key in color_range.keys():
        if(key != "白色"):
            pixel_count = np.sum(cv2.inRange(img, np.float32(color_range[key][0]), np.float32(color_range[key][1])))
            color_list.append([key, pixel_count])

    color_list.sort(key = lambda x: x[1], reverse = True)
    color = color_list[0][0][:2]
    
    return color 


# copy img into ./Results directory 
# based on color cluster
def classifyImg(img_path, color_dir):
    if not os.path.exists(color_dir):
        os.mkdir(color_dir)
    shutil.copy(img_path, color_dir)



if __name__ == "__main__":
    # create new folder under the cwd
    # if the folder has already existed
    # DELETE THE FOLDER AND CREATE AN EMPTY ONE!
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir) 
    os.mkdir(result_dir)

    for root, dirs, files in os.walk(img_dir):
        for file in files:
            # output the color catefory for each imgs (type: string)
            major_color = majoColor_inrange(os.path.join(root, file))
            # copy imgs to its corresponding categories in ./Results diretory
            classifyImg(os.path.join(root, file), os.path.join(result_dir, major_color))








