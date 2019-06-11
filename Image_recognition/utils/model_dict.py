import torch as t


def save_oplaus():
    state_dict = {}
    checkpoint = t.load('../checkpoint/EfficientNet.pth.tar')
    state_dict['state_dict'] = checkpoint['state_dict']
    t.save(state_dict, '/opt/checkpoint/EfficientNet.pth')


save_oplaus()
