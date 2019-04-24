echo '自动化训练'
prodir='/home/tian/Desktop/machine-learn/Image_recognition'
echo '进入项目'
cd $prodir
echo '切换环境'
source activate jiqi
echo '启动可视化'
python -m visdom.server &
echo '训练'
python main.py train
echo '测试'
python main.py test --load-path='checkpoints/ResNet152.pth.tar'
python utils/model_dict.py
echo 'scp传输'
scp /opt/checkpoint/ResNet152.pth thn@smallbird:~/
scp /opt/checkpoint/ResNet152.pth thn@birdwest:~/
