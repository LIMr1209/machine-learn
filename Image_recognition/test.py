import numpy as np
from config import opt
import torch as t
import models

memory_images = np.zeros(shape=(1, 200, 3, 32, 32), dtype=np.uint8)
print(memory_images)


def learn(**kwargs):
    # opt._parse(kwargs)
    # model = getattr(models, opt.model)()
    # if opt.load_model_path:
    #     checkpoint = t.load(opt.load_model_path)
    #     model.load_state_dict(checkpoint["state_dict"])  # 加载模型
    # model.to(opt.device)
    # feat = model.forward(t.from_numpy(np.zeros(shape=(1, 3, 224, 224))).float().cuda())
    # dim = np.shape(feat.cpu().data.numpy())[-1]  # 特征维度 (多少类)
    # print(dim)

    a = t.randn(5, 5)
    b = t.randn(5)
    print(a[b])


if __name__ == '__main__':
    learn()
