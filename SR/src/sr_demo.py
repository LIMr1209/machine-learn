import imageio
import model
import torch
from option import args
import utility
from data import common


# 生成高分辨率图片

def prepare(self, *args):
    device = torch.device('cpu' if self.args.cpu else 'cuda')

    def _prepare(tensor):
        if self.args.precision == 'half': tensor = tensor.half()
        return tensor.to(device)

    return [_prepare(a) for a in args]


# demo_gen == True
_model = model.Model(args)

load_from = torch.load('../experiment/WDSR_B_BIX4/model/model_best.pt')
_model.load_state_dict(load_from, strict=False)

torch.set_grad_enabled(False)
_model.eval()

for idx_scale, scale in enumerate(args.scale):
    lr = imageio.imread('../test/0810x4.png')
    lr, = common.set_channel(lr, n_channels=args.n_colors)
    lr_t, = common.np2Tensor(lr, rgb_range=args.rgb_range)
    lr_t = lr_t.to(torch.device('cuda'))
    lr_t = lr_t.view(1, 3, 384, 510)
    sr = _model(lr_t, idx_scale)
    sr = utility.quantize(sr, args.rgb_range)
    normalized = sr[0].mul(255 / args.rgb_range)
    tensor_cpu = normalized.byte().permute(1, 2, 0).cpu()
    filename = '../test/0810x4_test.png'
    imageio.imwrite(filename, tensor_cpu.numpy())

torch.set_grad_enabled(True)
