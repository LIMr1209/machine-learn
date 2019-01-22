## 环境准备

- 本程序需要安装[PyTorch](https://pytorch.org/) 本人安装总结[地址](https://blog.csdn.net/qq_41654985/article/details/86599016)
- 还需要通过`pip install -r requirements.txt` 安装其它依赖

## 用法
如果想要使用visdom可视化，请先运行`python2 -m visdom.server`启动visdom服务

## 训练
然后使用如下命令启动训练：

```
# 在gpu0上训练,并把可视化结果保存在visdom 的classifier env上
python main.py train --train-data-root=./data/train --use-gpu --env=classifier
```


详细的使用命令 可使用
```
python main.py help
```

## 测试

```
python main.py test --data-root=./data/test  --batch-size=256 --load-path='checkpoints/squeezenet.pth.tar'
```

## 识别

```
python main.py recognition --url='图片地址' --load-path='checkpoints/squeezenet.pth.tar'
```
