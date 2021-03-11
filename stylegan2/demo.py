import io
import json
import time

import dnnlib
import dnnlib.tflib as tflib
import numpy as np
from PIL import Image
import pickle
import requests
from qiniu import Auth, put_file, put_data, put_stream, etag, BucketManager
from bson import ObjectId

network_path = "network_pkl"
truncation_psi = 0.5



stream = open(network_path, 'rb')
tflib.init_tf()
with stream:
    G, D, Gs = pickle.load(stream, encoding='latin1')
for i in range(30000):
    noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]

    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_kwargs.randomize_noise = False
    Gs_kwargs.truncation_psi = truncation_psi

    rnd = np.random.RandomState(i)
    z = rnd.randn(1, *Gs.input_shape[1:]) # [minibatch, component]
    tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars}) # [height, width]
    images = Gs.run(z, None, **Gs_kwargs) # [minibatch, height, width, channel]
    img = Image.fromarray(images[0], 'RGB')

    access_key = "ERh7qjVSy0v42bQ0fftrFeKYZG39XbzRlaJO4NFy"
    secret_key = "r-NUrKsnRBEwTQxbLONVrK9tPuncXyHmcq4BkSc7"
    bucket_name = "opalus"
    key = "%s/%s/%s/%s" % (
        bucket_name,
        'stylegan',
        time.strftime("%y%m%d"),
        ObjectId().__str__(),
    )

    imgByteArr = io.BytesIO()
    img.save(imgByteArr, format='JPEG')
    imgByteArr = imgByteArr.getvalue()

    try:
        # 构建鉴权对象
        q = Auth(access_key, secret_key)
        # 生成上传 Token，可以指定过期时间等
        token = q.upload_token(bucket_name, key, 600)
        ret, info = put_data(token, key, imgByteArr)
        if not info.status_code == 200:
            print(info.error)
            break
        opalus_url = "https://opalus.d3ingo.com/api/image/submit"
        data = {
            'path': key,
            'kind': 1
        }
        res = requests.post(opalus_url, data)
        res_data = json.loads(res.content)
        if res_data['code'] != 0:
            print(res_data['message'])
            break
    except Exception as e:
        print(e)
        break