from config import opt
import torch as t

state_dict = {}
checkpoint = t.load(opt.load_model_path)
if 'quantize' in opt.load_model_path:
    state_dict['quantizer_metadata'] = checkpoint['quantizer_metadata']
state_dict['state_dict'] = checkpoint['state_dict']
if 'quantize' in opt.load_model_path:
    t.save(state_dict, '/opt/checkpoint/' + opt.model + '_quantizer.pth')
else:
    t.save(state_dict, '/opt/checkpoint/' + opt.model + '.pth')
