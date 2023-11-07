from model import Generator
import torch
from torchvision import utils
import argparse

device = "cuda"
size = 256
truncation = 1
truncation_mean = 4096
ckpt = "checkpoint/090000.pt"
channel_multiplier = 2
factor = "factor.pt"
torch.set_grad_enabled(False)
parser = argparse.ArgumentParser(description="Demo")
parser.add_argument(
    "--torch_seed",
    type=int,
    default=0,
    help="seed for generating random numbers",
)
parser.add_argument(
    "--func",
    type=str,
    default="generate",
    help="执行函数",
)

parser.add_argument(
    "-i", "--index", type=int, default=0, help="index of eigenvector"
)
parser.add_argument(
    "-d",
    "--degree",
    type=float,
    default=5,
    help="scalar factors for moving latent vectors along eigenvector",
)
args = parser.parse_args()

if args.torch_seed > 0:
    torch.manual_seed(args.torch_seed)
g_ema = Generator(size, 512, 8, channel_multiplier=2).to(device)
checkpoint = torch.load(ckpt, map_location=lambda storage, loc: storage)

g_ema.load_state_dict(checkpoint["g_ema"])

g_ema.eval()
sample_z = torch.randn(1, 512, device=device)


def generate():
    sample, _ = g_ema(
        [sample_z], truncation=truncation, truncation_latent=None
    )

    utils.save_image(
        sample,
        f"demo/{args.torch_seed}.png",
        nrow=1,
        normalize=True,
        range=(-1, 1),
    )


def apply_factor():
    eigvec = torch.load(factor)["eigvec"].to(device)
    latent = g_ema.get_latent(sample_z)
    trunc = g_ema.mean_latent(truncation_mean)

    direction = args.degree * eigvec[:, args.index].unsqueeze(0)

    sample, _ = g_ema(
        [latent + direction],
        truncation=truncation,
        truncation_latent=trunc,
        input_is_latent=True,
    )
    utils.save_image(
        sample,
        f"demo/{args.torch_seed}_{args.index}_{args.degree}.png",
        nrow=1,
        normalize=True,
        range=(-1, 1),
    )


if __name__ == '__main__':
    eval(args.func)()
