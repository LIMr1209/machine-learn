from aip import AipImageClassify

""" 你的 APPID AK SK """
APP_ID = '15180965'
API_KEY = 'Gjw3RzhDcMSS8RESUEiVNWkH'
SECRET_KEY = 'pZVtDe5Z74cA2ropdsI3s3rGMrFmXH9N'

client = AipImageClassify(APP_ID, API_KEY, SECRET_KEY)

""" 读取图片 """


def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()


image = get_file_content('images/1.jpg')

""" 调用图像主体检测 """
client.objectDetect(image)

""" 如果有可选参数 """
options = {}
options["with_face"] = 0

""" 带参数调用图像主体检测 """
response = client.objectDetect(image, options)
a = 1
