import torch.onnx
import torchvision
import torch as t

# onnx 实现多个机器学习框架转化模型
dummy_input = t.randn(10, 3, 224, 224).cuda()
model = torchvision.models.alexnet(pretrained=True).cuda()
torch.onnx.export(model, dummy_input, "alexnet.proto", verbose=True)  # 转换onnx 开放神经网络交换

import onnx
import numpy as np
from onnx_tf.backend import prepare

model = onnx.load('alexnet.proto')  # onnx 加载转化后的onnx模型
tf_rep = prepare(model)  # onnx 模型转化为tensorflow 模型

img = np.load("11.jpeg")
output = tf_rep.run(img.reshape([10, 3, 224, 224]))

print("outpu mat: \n", output)
print("The digit is classified as ", np.argmax(output))

import tensorflow as tf

with tf.Session() as persisted_sess:
    print("load graph")
    persisted_sess.graph.as_default()
    tf.import_graph_def(tf_rep.predict_net.graph.as_graph_def(), name='')
    inp = persisted_sess.graph.get_tensor_by_name(
        tf_rep.predict_net.tensor_dict[tf_rep.predict_net.external_input[0]].name
    )
    out = persisted_sess.graph.get_tensor_by_name(
        tf_rep.predict_net.tensor_dict[tf_rep.predict_net.external_output[0]].name
    )
    res = persisted_sess.run(out, {inp: img.reshape([10, 3, 224, 224])})
    print(res)
    print("The digit is classified as ", np.argmax(res))

tf_rep.export_graph('tf.pb')
