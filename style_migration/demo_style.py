from models import networks
from torchvision import transforms as T
from PIL import Image
import torch

from util.util import tensor2im

transforms = T.Compose([
    T.Resize([256, 256], Image.BICUBIC),
    T.RandomCrop(256),
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


def __patch_instance_norm_state_dict(state_dict, module, keys, i=0):
    """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
    key = keys[i]
    if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
        if module.__class__.__name__.startswith('InstanceNorm') and \
                (key == 'running_mean' or key == 'running_var'):
            if getattr(module, key) is None:
                state_dict.pop('.'.join(keys))
        if module.__class__.__name__.startswith('InstanceNorm') and \
                (key == 'num_batches_tracked'):
            state_dict.pop('.'.join(keys))
    else:
        __patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)


def load_img(path):
    img = Image.open(path).convert('RGB')
    input = transforms(img)
    input = input.view(1, 3, 256, 256).to(torch.device('cuda'))
    return input


def load_model(path):
    model = networks.define_G(3, 3, 64, 'resnet_9blocks', 'instance', False, 'normal', 0.02, [0])
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    state_dict = torch.load(path)
    if hasattr(state_dict, '_metadata'):
        del state_dict._metadata
    for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
        __patch_instance_norm_state_dict(state_dict, model, key.split('.'))
    model.load_state_dict(state_dict)
    model.eval()
    return model


if __name__ == '__main__':
    model = load_model('./checkpoints/horse2zebra_pretrained/latest_net_G.pth')
    input = load_img('./test/test.jpg')
    torch.set_grad_enabled(False)
    output = model(input)
    result = tensor2im(output)
    img_new = Image.fromarray(result)
    img_new.save('./test/test_new.jpg')
    torch.set_grad_enabled(True)
