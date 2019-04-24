from config import opt
import torch as t

state_dict = {}
checkpoint = t.load(opt.load_model_path)
state_dict['state_dict'] = checkpoint['state_dict']
t.save(state_dict, '/opt/checkpoint/' + opt.model + '.pth')
