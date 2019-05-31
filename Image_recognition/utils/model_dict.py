import torch as t


def save_oplaus():
    state_dict = {}
    checkpoint = t.load('../checkpoint/ResNet152.pth.tar')
    state_dict['state_dict'] = checkpoint['state_dict']
    t.save(state_dict, '/opt/checkpoint/ResNet152.pth')


save_oplaus()
