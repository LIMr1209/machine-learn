from io import BytesIO
from torchvision import transforms as T
import requests
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # 图片文件允许截断
from config import opt

normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])

transforms = T.Compose([
    T.Resize(opt.image_size),
    T.ToTensor(),
    normalize
])


def image_loader(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36'}
    response = requests.get(url, headers=headers)
    image = Image.open(BytesIO(response.content)).convert('RGB')

    return transforms(image)
