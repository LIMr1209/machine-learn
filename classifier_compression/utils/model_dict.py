import torch as t
from resnet152 import ResNet152

model = ResNet152()
checkpoint = t.load('/home/tian/Desktop/machine-learn/classifier_compression/logs/2019.03.15-163456/checkpoint.pth.tar')
model.load_state_dict(checkpoint["state_dict"])
t.save(model.state_dict(), '/opt/checkpoint/' + model.model_name + '.pt')
