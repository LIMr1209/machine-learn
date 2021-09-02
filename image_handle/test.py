import os
from PIL import Image

# 转换格式
IMG_EXP = ".png"
CURRENT_PATH = 'C:\\Users\\aaa10\\Desktop\\webp'
# 获取最高所有文件
cur_all_files = os.listdir(CURRENT_PATH)
# 转换列表
imgList = []


# 遍历文件夹，储存webp格式的路径到列表内
def findFileForImage(filePath):
    child_all_files = os.listdir(filePath)
    for child_file_name in child_all_files:
        sPath = os.path.join(filePath, child_file_name)
        if os.path.isdir(sPath):
            findFileForImage(sPath)
        n, e = os.path.splitext(child_file_name)
        if e.lower() == ".webp":
            imgList.append(os.path.join(filePath, n))


# 检索目录下所有的webp文件，如果是文件夹则继续向下检索
for file_name in cur_all_files:
    nPath = os.path.join(CURRENT_PATH, file_name)
    # 文件夹
    if os.path.isdir(nPath):
        findFileForImage(nPath)
        continue
    # 储存
    name, ext = os.path.splitext(file_name)
    if ext.lower() == ".webp":
        imgList.append(os.path.join(CURRENT_PATH, name))


# 转换图片
def convertImage():
    for webpPath in imgList:
        print(webpPath)

        # 打开图片并赋值一份新的图片
        img = Image.open(webpPath + ".webp")
        img.load()
        # 将赋值的图片修改后缀保存在原路径
        img.save(webpPath + IMG_EXP)
        # 删除原webp图
        os.remove(webpPath + ".webp")


# 执行
convertImage()