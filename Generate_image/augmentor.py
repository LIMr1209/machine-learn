import Augmentor  # 图像增强 图像预处理  https://github.com/mdbloice/Augmentor     https://augmentor.readthedocs.io/en/master/
import torchvision

p = Augmentor.Pipeline("/home/tian/Desktop/spiders/design/design/spiders/image_test/dog")

p.rotate90(probability=0.5)
p.rotate270(probability=0.5)
p.flip_left_right(probability=0.8)
p.flip_top_bottom(probability=0.3)
p.crop_random(probability=1, percentage_area=0.5)
p.resize(probability=1.0, width=224, height=224)

transforms = torchvision.transforms.Compose([
    p.torch_transform(),
    torchvision.transforms.ToTensor(),
])
