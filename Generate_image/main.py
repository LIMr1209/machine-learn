import os
import torch as t
import torchvision as tv
import tqdm
from models import NetG, NetD
from torchnet.meter import AverageValueMeter
from config import opt


def train(**kwargs):
    opt._parse(kwargs)
    if opt.vis:
        from utils.visualize import Visualizer
        vis = Visualizer(opt.env)

    # 数据
    transforms = tv.transforms.Compose([
        tv.transforms.Resize(opt.image_size),
        tv.transforms.CenterCrop(opt.image_size),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = tv.datasets.ImageFolder(opt.data_path, transform=transforms)
    dataloader = t.utils.data.DataLoader(dataset,
                                         batch_size=opt.batch_size,
                                         shuffle=True,
                                         num_workers=opt.num_workers,
                                         drop_last=True  # 最后一个数据集不满batch_size  将被遗弃
                                         )

    # 网络
    netg, netd = NetG(opt), NetD(opt)
    map_location = lambda storage, loc: storage
    if opt.netd_path:
        netd.load_state_dict(t.load(opt.netd_path, map_location=map_location))
    if opt.netg_path:
        netg.load_state_dict(t.load(opt.netg_path, map_location=map_location))
    netd.to(opt.device)
    netg.to(opt.device)

    # 定义优化器和损失
    optimizer_g = t.optim.Adam(netg.parameters(), opt.lr1, betas=(opt.beta1, 0.999))
    optimizer_d = t.optim.Adam(netd.parameters(), opt.lr2, betas=(opt.beta1, 0.999))
    criterion = t.nn.BCELoss().to(opt.device)

    # 真图片label为1，假图片label为0
    # noises为生成网络的输入
    true_labels = t.ones(opt.batch_size).to(opt.device)  # 真
    fake_labels = t.zeros(opt.batch_size).to(opt.device)  # 假
    fix_noises = t.randn(opt.batch_size, opt.nz, 1, 1).to(opt.device)
    noises = t.randn(opt.batch_size, opt.nz, 1, 1).to(opt.device)

    errord_meter = AverageValueMeter()
    errorg_meter = AverageValueMeter()

    epochs = range(opt.max_epoch)
    for epoch in iter(epochs):
        for ii, (img, _) in tqdm.tqdm(enumerate(dataloader)):
            real_img = img.to(opt.device)

            if ii % opt.d_every == 0:
                # 训练判别器
                optimizer_d.zero_grad()
                ## 尽可能的把真图片判别为正确
                output = netd(real_img)
                error_d_real = criterion(output, true_labels)
                error_d_real.backward()

                ## 尽可能把假图片判别为错误
                noises.data.copy_(t.randn(opt.batch_size, opt.nz, 1, 1))
                fake_img = netg(noises).detach()  # 根据噪声生成假图
                output = netd(fake_img)
                error_d_fake = criterion(output, fake_labels)
                error_d_fake.backward()
                optimizer_d.step()

                error_d = error_d_fake + error_d_real

                errord_meter.add(error_d.item())

            if ii % opt.g_every == 0:
                # 训练生成器
                optimizer_g.zero_grad()
                noises.data.copy_(t.randn(opt.batch_size, opt.nz, 1, 1))
                fake_img = netg(noises)
                output = netd(fake_img)
                error_g = criterion(output, true_labels)
                error_g.backward()
                optimizer_g.step()
                errorg_meter.add(error_g.item())

            if opt.vis and ii % opt.plot_every == opt.plot_every - 1:
                ## 可视化
                fix_fake_imgs = netg(fix_noises)
                vis.images(fix_fake_imgs.detach().cpu().numpy()[:64] * 0.5 + 0.5, win='fixfake')
                vis.images(real_img.data.cpu().numpy()[:64] * 0.5 + 0.5, win='real')
                vis.plot('errord', errord_meter.value()[0])
                vis.plot('errorg', errorg_meter.value()[0])

        if (epoch + 1) % opt.save_every == 0:
            # 保存模型、图片
            tv.utils.save_image(fix_fake_imgs.data[:64], '%s/%s.png' % (opt.save_path, epoch), normalize=True,
                                range=(-1, 1))
            tag = [i for i in os.listdir('./data') if os.path.isdir('./data/' + i)][0]
            t.save(netd.state_dict(), 'checkpoints/%s_d_%s.pth' % (tag, epoch))
            t.save(netg.state_dict(), 'checkpoints/%s_g_%s.pth' % (tag, epoch))
            errord_meter.reset()
            errorg_meter.reset()


def generate(**kwargs):
    """
    随机生成图像，并根据netd的分数选择较好的
    """
    with t.no_grad():
        opt._parse(kwargs)
        netg, netd = NetG(opt).eval(), NetD(opt).eval()
        noises = t.randn(opt.gen_search_num, opt.nz, 1, 1).normal_(opt.gen_mean, opt.gen_std)
        noises = noises.to(opt._parse(kwargs))

        map_location = lambda storage, loc: storage
        netd.load_state_dict(t.load(opt.netd_path, map_location=map_location))
        netg.load_state_dict(t.load(opt.netg_path, map_location=map_location))
        netd.to(opt._parse(kwargs))
        netg.to(opt._parse(kwargs))

        # 生成图片，并计算图片在判别器的分数
        fake_img = netg(noises)
        scores = netd(fake_img).detach()

        # 挑选最好的某几张
        indexs = scores.topk(opt.gen_num)[1]
        result = []
        for ii in indexs:
            result.append(fake_img.data[ii])
        # 保存图片
        tv.utils.save_image(t.stack(result), opt.gen_img, normalize=True, range=(-1, 1))


if __name__ == '__main__':
    import fire

    fire.Fire()
