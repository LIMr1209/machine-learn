import argparse

import torch
from torchvision import utils

from model import Generator


if __name__ == "__main__":
    torch.set_grad_enabled(False)

    parser = argparse.ArgumentParser(description="Apply closed form factorization")

    parser.add_argument(
        "-e_i", "--end_index", type=int, default=0, help="index of eigenvector"
    )

    parser.add_argument(
        "-i_num", "--index_num", type=int, default=1, help="number index of eigenvector"
    )
    parser.add_argument(
        "-s_i", "--start_index", type=int, default=1, help="number index of eigenvector"
    )
    parser.add_argument(
        "-d",
        "--degree",
        type=float,
        default=5,
        help="scalar factors for moving latent vectors along eigenvector",
    )
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help='channel multiplier factor. config-f = 2, else = 1',
    )
    parser.add_argument(
        "-d_num",
        "--degree_num",
        type=int,
        default=3,
        help="number of scalar factors for moving latent vectors along eigenvector",
    )
    parser.add_argument("--ckpt", type=str, required=True, help="stylegan2 checkpoints")
    parser.add_argument(
        "--size", type=int, default=256, help="output image size of the generator"
    )
    parser.add_argument(
        "-n", "--n_sample", type=int, default=7, help="number of samples created"
    )
    parser.add_argument(
        "--truncation", type=float, default=0.7, help="truncation factor"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="device to run the model"
    )
    parser.add_argument(
        "--out_prefix",
        type=str,
        default="factor",
        help="filename prefix to result samples",
    )
    parser.add_argument(
        "factor",
        type=str,
        help="name of the closed form factorization result factor file",
    )
    parser.add_argument(
        "--torch_seed",
        type=int,
        default=0,
        help="seed for generating random numbers",
    )

    args = parser.parse_args()

    eigvec = torch.load(args.factor)["eigvec"].to(args.device)
    ckpt = torch.load(args.ckpt)
    g = Generator(args.size, 512, 8, channel_multiplier=args.channel_multiplier).to(args.device)
    g.load_state_dict(ckpt["g_ema"], strict=False)

    trunc = g.mean_latent(4096)

    if args.torch_seed > 0:
        torch.manual_seed(args.torch_seed)
    latent = torch.randn(args.n_sample, 512, device=args.device)
    latent = g.get_latent(latent)

    for i in range(args.start_index, args.end_index, args.index_num):
        direction = eigvec[:, i].unsqueeze(0)
        img_dict = dict()
        for u in torch.linspace(- args.degree, args.degree, args.degree_num):
            img_batch, _ = g(
                [latent + u * direction],
                truncation=args.truncation,
                truncation_latent=trunc,
                input_is_latent=True,
            )

            for j in range(img_batch.shape[0]):

                img = img_batch[j].unsqueeze(0)

                try:
                    img_dict[j].append(img)
                except KeyError:
                    img_dict[j] = [img]

        img_list = [
            torch.cat(img_dict[j], 0)
            for j in range(args.n_sample)
        ]

        grid = utils.save_image(
            torch.cat(img_list, 0),
            f"test_sample/{args.out_prefix}_index-{i:02d}_degree-{args.degree}.jpg",
            normalize=True,
            range=(-1, 1),
            nrow=args.degree_num,
        )

