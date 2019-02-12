import os
import shutil
import numpy as np
import cv2

for i in os.listdir('./Water_cup'):
    try:
        image = cv2.imdecode(np.fromfile('./Water_cup'+'/'+i, dtype=np.uint8), -1)
        print(image.shape)
    except:
        os.remove('./Water_cup'+'/'+i)
        print('删除成功')
