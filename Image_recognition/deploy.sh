#!/usr/bin/zsh

# 加载python环境
current_path=$(cd $(dirname $0);pwd)
source activate jiqi

case "$1" in
start)
	pids=`ps -ef | grep 'tensorboard'| grep -v 'grep' | awk '{print $2}'`
	if [ $pids ];then
		echo "tensorboard is runing  http://tian-orc:6006"
	else
		echo "strat tensorboard  http://tian-orc:6006"
        nohup tensorboard --logdir=runs &
	fi
	echo '训练'
    python main.py train
    echo '测试'
    python main.py test --load-path='checkpoints/ResNet152.pth.tar'
    python utils/model_dict.py

;;
stop)
	pids=`ps -ef | grep 'deploy.sh start'| grep -v 'grep' | awk '{print $2}'`
    if [ $pids ];then
		kill -9 $pids
		ps -ef | grep 'python'| grep -v 'grep' | awk '{print $2}' | xargs kill -9
		echo "Stop  [OK]"
	else
		echo "AI not start"
	fi
;;
*)
    echo "Usages: ./deploy.sh [start|stop]"
;;
esac
