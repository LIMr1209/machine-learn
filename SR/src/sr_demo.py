import imageio
import model
import torch
from option import args
import utility
from data import common

args.demo_gen = True
args.n_feats = 128
args.block_feats = 512
args.n_resblocks = 32
args.res_scale = 0.1
args.scale = [4]
args.model = 'wdsr_b'


# 生成高分辨率图片
def load_model(path):
    _model = model.Model(args)
    load_from = torch.load(path)
    _model.model.load_state_dict(load_from)
    _model.eval()
    return _model


def load_img(path):
    lr = imageio.imread(path)
    lr, = common.set_channel(lr, n_channels=args.n_colors)
    lr_t, = common.np2Tensor(lr, rgb_range=args.rgb_range)
    lr_t = lr_t.view(1, 3, 384, 510).to(torch.device('cuda'))
    return lr_t


def save_img(output, path):
    output = utility.quantize(output, args.rgb_range)
    normalized = output[0].mul(255 / args.rgb_range)
    tensor_cpu = normalized.byte().permute(1, 2, 0).cpu()
    filename = path % scale
    imageio.imwrite(filename, tensor_cpu.numpy())


if __name__ == '__main__':
    _model = load_model('../experiment/WDSR_B_BIX4/model/model_best.pt')
    torch.set_grad_enabled(False)
    for idx_scale, scale in enumerate(args.scale):
        input = load_img('../test/0810.png')
        output = _model(input, idx_scale)
        save_img(output, '../test/0810x%s.png')
    torch.set_grad_enabled(True)
