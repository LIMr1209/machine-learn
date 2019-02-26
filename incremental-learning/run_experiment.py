from __future__ import print_function

import argparse
import logging
import time
import torch
import torch.utils.data as td
import data_handler
import experiment as ex
import model
import trainer

parser = argparse.ArgumentParser(description='iCarl2.0')
parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                    help='input batch size for training (default: 16)')
parser.add_argument('--lr', type=float, default=2.0, metavar='LR',
                    help='learning rate (default: 2.0). Note that lr is decayed by args.gamma parameter args.schedule ')
parser.add_argument('--schedule', type=int, nargs='+', default=[5, 7, 9],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--gammas', type=float, nargs='+', default=[0.2, 0.2, 0.2],
                    help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--random-init', action='store_true', default=False,
                    help='Initialize model for next increment using previous weights if false and random weights otherwise')
# 如果为假，则使用前一个权重初始化下一个增量的模型，否则使用随机权重
parser.add_argument('--no-distill', action='store_true', default=False,
                    help='disable distillation loss and only uses the cross entropy loss. See "Distilling Knowledge in Neural Networks" by Hinton et.al for details')
# 禁用蒸馏损失，只使用交叉熵损失。详见Hinton等人的“提取神经网络中的知识”
parser.add_argument('--no-random', action='store_true', default=False,
                    help='Disable random shuffling of classes')
parser.add_argument('--no-herding', action='store_true', default=False,
                    help='Disable herding algorithm and do random instance selection instead')
parser.add_argument('--seeds', type=int, nargs='+', default=[23423],
                    help='Seeds values to be used; seed introduces randomness by changing order of classes')
parser.add_argument('--log-interval', type=int, default=5, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model-type', default="resnet152",
                    help='model type to be used. Example : resnet32, resnet20, test')
parser.add_argument('--name', default="noname",
                    help='Name of the experiment')
parser.add_argument('--outputDir', default="../",
                    help='Directory to store the results; a new folder "DDMMYYYY" will be created '
                         'in the specified directory to save the results.')
parser.add_argument('--upsampling', action='store_true', default=False,
                    help='Do not do upsampling.')
parser.add_argument('--unstructured-size', type=int, default=0,
                    help='Leftover parameter of an unreported experiment; leave it at 0')
parser.add_argument('--alphas', type=float, nargs='+', default=[1.0],
                    help='Weight given to new classes vs old classes in the loss; high value of alpha will increase perfomance on new classes at the expense of older classes. Dynamic threshold moving makes the system more robust to changes in this parameter')
parser.add_argument('--decay', type=float, default=0.00005, help='Weight decay (L2 penalty).')
parser.add_argument('--step-size', type=int, default=3,
                    help='How many classes to add in each increment')  # 每个增量中要添加多少类
parser.add_argument('--T', type=float, default=1, help='Tempreture used for softening the targets')  # 软化目标所用的温度
parser.add_argument('--memory-budgets', type=int, nargs='+', default=[2000],
                    help='How many images can we store at max. 0 will result in fine-tuning')
parser.add_argument('--epochs-class', type=int, default=10, help='Number of epochs for each increment')
parser.add_argument('--dataset', default="custom", help='Dataset to be used; example CIFAR, MNIST')
parser.add_argument('--lwf', action='store_true', default=False,
                    help='Use learning without forgetting. Ignores memory-budget '
                         '("Learning with Forgetting," Zhizhong Li, Derek Hoiem)')
parser.add_argument('--no-nl', action='store_true', default=False,
                    help='No Normal Loss. Only uses the distillation loss to train the new model on old classes (Normal loss is used for new classes however')
# 无正常损失。只使用蒸馏损失来训练旧类的新模型（但是新类使用正常损失)
parser.add_argument('--old', action='store_true', default=False,
                    help='Preloading the old model default false')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


def add_log(my_experiment):
    # Adding support for logging. A .log is generated with all the logs.
    # Logs are also stored in a temp file one directory
    # 正在添加对日志记录的支持。日志与所有日志一起生成。日志也存储在临时文件目录中
    # before the code repository
    logger = logging.getLogger('iCARL')
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(my_experiment.path + ".log")
    fh.setLevel(logging.DEBUG)

    fh2 = logging.FileHandler("./log/temp.log")
    fh2.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    fh2.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(fh2)
    logger.addHandler(ch)
    return logger


dataset = data_handler.DatasetFactory.get_dataset(args.dataset)

# Checks to make sure parameters are sane  检查以确保参数正常
if args.step_size < 2:
    print("Step size of 1 will result in no learning;")  # 参数step_size 为1 无法学习
    assert False

# Run an experiment corresponding to every seed value  运行与每个种子值对应的实验
args.seed = args.seeds[0]
# Run an experiment corresponding to every alpha value  运行与每个alpha值对应的实验
args.alpha = args.alphas[0]
# Run an experiment corresponding to every memory budget  根据每个内存预算运行一个实验
args.memory_budget = args.memory_budgets[0]
# In LwF, memory_budget is 0 (See the paper "Learning without Forgetting" for details).
if args.lwf:
    args.memory_budget = 0
# Fix the seed.
torch.manual_seed(args.seed)  # 随机数种子,当使用随机数时,关闭进程后再次生成和上次得一样
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Loader used for training data  # 训练数据加载器
train_dataset_loader = data_handler.IncrementalLoader(dataset.train_data.train_data,
                                                      dataset.train_data.train_labels,
                                                      dataset.labels_per_class_train,
                                                      dataset.classes, [1, 2],
                                                      transform=dataset.train_transform,
                                                      cuda=args.cuda, oversampling=args.upsampling,
                                                      )
# Special loader use to compute ideal NMC; i.e, NMC that using all the data points to compute the mean embedding
train_dataset_loader_nmc = data_handler.IncrementalLoader(dataset.train_data.train_data,
                                                          dataset.train_data.train_labels,
                                                          dataset.labels_per_class_train,
                                                          dataset.classes, [1, 2],
                                                          transform=dataset.train_transform,
                                                          cuda=args.cuda, oversampling=args.upsampling,
                                                          )
# Loader for test data. # 测试数据加载器
test_dataset_loader = data_handler.IncrementalLoader(dataset.test_data.test_data,
                                                     dataset.test_data.test_labels,
                                                     dataset.labels_per_class_test, dataset.classes,
                                                     [1, 2], transform=dataset.test_transform,
                                                     cuda=args.cuda, oversampling=args.upsampling
                                                     )

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

# Iterator to iterate over training data. # 迭代器迭代训练数据。
train_iterator = torch.utils.data.DataLoader(
    train_dataset_loader,
    batch_size=args.batch_size, shuffle=True, **kwargs
)
# Iterator to iterate over all training data (Equivalent to memory-budget = infitie
train_iterator_nmc = torch.utils.data.DataLoader(
    train_dataset_loader_nmc,
    batch_size=args.batch_size, shuffle=True, **kwargs
)
# Iterator to iterate over test data  # 迭代器迭代测试数据。
test_iterator = torch.utils.data.DataLoader(
    test_dataset_loader,
    batch_size=args.batch_size, shuffle=True, **kwargs
)

# Get the required model  得到模型
myModel = model.ModelFactory.get_model(args.model_type, args.dataset)
if args.old:
    myModel.load_state_dict(torch.load('./checkpoint/'))
if args.cuda:
    myModel.to(torch.device('cuda'))

# Define an experiment. 定义一个实验
my_experiment = ex.experiment(args.name, args)

logger = add_log(my_experiment)

# Define the optimizer used in the experiment  SGD 优化器
optimizer = torch.optim.SGD(myModel.parameters(), args.lr, momentum=args.momentum,
                            weight_decay=args.decay, nesterov=True)

# Trainer object used for training  # 训练对象类
my_trainer = trainer.Trainer(train_iterator, test_iterator, dataset, myModel, args, optimizer,
                             train_iterator_nmc)

# Parameters for storing the results  存储结果参数
x = []
y = []
train_y = []
y1 = []
y_scaled = []
y_grad_scaled = []
nmc_ideal_cum = []

# Initilize the evaluators used to measure the performance of the system.
# 初始化用于测量系统性能的评估器
nmc = trainer.EvaluatorFactory.get_evaluator("nmc", args.cuda)  # nmc 返回近距离平均评估器
nmc_ideal = trainer.EvaluatorFactory.get_evaluator("nmc", args.cuda)  # nmc 返回近距离平均评估器
t_classifier = trainer.EvaluatorFactory.get_evaluator("trainedClassifier", args.cuda)  # 训练分类器 返回softmax 分类器

# Loop that incrementally adds more and more classes  # 循环，递增地添加越来越多的类
for class_group in range(0, dataset.classes, args.step_size):
    print("SEED:", args.seed, "MEMORY_BUDGET:", args.memory_budget, "CLASS_GROUP:", class_group)
    # Add new classes to the train, train_nmc, and test iterator
    my_trainer.increment_classes(class_group)
    my_trainer.update_frozen_model()  # 更新冻结模型

    # Running epochs_class epochs
    for epoch in range(args.epochs_class):
        my_trainer.update_lr(epoch)  # 更新学习效率
        my_trainer.train(epoch)
        # print(my_trainer.threshold)
        if epoch % args.log_interval == (args.log_interval - 1):
            tError = t_classifier.evaluate(my_trainer.model, train_iterator)
            logger.debug("*********CURRENT EPOCH********** : %d", epoch)
            logger.debug("Train Classifier: %0.2f", tError)
            logger.debug("Test Classifier: %0.2f", t_classifier.evaluate(my_trainer.model, test_iterator))
            logger.debug("Test Classifier Scaled: %0.2f",
                         t_classifier.evaluate(my_trainer.model, test_iterator,
                                               my_trainer.dynamic_threshold, False,
                                               my_trainer.older_classes, args.step_size))
            logger.info("Test Classifier Grad Scaled: %0.2f",
                        t_classifier.evaluate(my_trainer.model, test_iterator,
                                              my_trainer.gradient_threshold_unreported_experiment, False,
                                              my_trainer.older_classes, args.step_size))

    # Evaluate the learned classifier  评估学习的分类器
    test_eval = t_classifier.evaluate(my_trainer.model, test_iterator)
    test_scaled_eval = t_classifier.evaluate(my_trainer.model, test_iterator, my_trainer.dynamic_threshold, False,
                                             my_trainer.older_classes, args.step_size)
    test_grad_scaled_eval = t_classifier.evaluate(my_trainer.model, test_iterator,
                                                  my_trainer.gradient_threshold_unreported_experiment, False,
                                                  my_trainer.older_classes, args.step_size)
    logger.info("Test Classifier Final: %0.2f", test_eval)
    logger.info("Test Classifier Final Scaled: %0.2f", test_scaled_eval)
    logger.info("Test Classifier Final Grad Scaled: %0.2f", test_grad_scaled_eval)
    y_grad_scaled.append(test_scaled_eval)
    y_scaled.append(test_scaled_eval)
    y1.append(test_eval)

    # Update means using the train iterator; this is iCaRL case  使用训练迭代器更新平均数中间数
    nmc.update_means(my_trainer.model, train_iterator, dataset.classes)
    # Update mean using all the data. This is equivalent to memory_budget = infinity 使用所有数据更新平均数中间数 这相当于内存预算=无穷大
    nmc_ideal.update_means(my_trainer.model, train_iterator_nmc, dataset.classes)
    # Compute the the nmc based classification results 计算基于NMC的分类结果
    tempTrain = t_classifier.evaluate(my_trainer.model, train_iterator)
    train_y.append(tempTrain)

    testY1 = nmc.evaluate(my_trainer.model, test_iterator, step_size=args.step_size, kMean=True)
    testY = nmc.evaluate(my_trainer.model, test_iterator)
    testY_ideal = nmc_ideal.evaluate(my_trainer.model, test_iterator)
    y.append(testY)
    nmc_ideal_cum.append(testY_ideal)

    # Compute confusion matrices of all three cases (Learned classifier, iCaRL, and ideal NMC)
    # 计算所有三种情况的混淆矩阵（学习分类器、iCarl和理想NMC）
    tcMatrix = t_classifier.get_confusion_matrix(my_trainer.model, test_iterator, dataset.classes)
    tcMatrix_scaled = t_classifier.get_confusion_matrix(my_trainer.model, test_iterator, dataset.classes,
                                                        my_trainer.dynamic_threshold,
                                                        my_trainer.older_classes,
                                                        args.step_size)
    tcMatrix_grad_scaled = t_classifier.get_confusion_matrix(my_trainer.model, test_iterator,
                                                             dataset.classes,
                                                             my_trainer.gradient_threshold_unreported_experiment,
                                                             my_trainer.older_classes,
                                                             args.step_size)
    nmcMatrix = nmc.get_confusion_matrix(my_trainer.model, test_iterator, dataset.classes)
    nmcMatrixIdeal = nmc_ideal.get_confusion_matrix(my_trainer.model, test_iterator, dataset.classes)
    tcMatrix_scaled_binning = t_classifier.get_confusion_matrix(my_trainer.model, test_iterator,
                                                                dataset.classes,
                                                                my_trainer.dynamic_threshold,
                                                                my_trainer.older_classes,
                                                                args.step_size, True)

    my_trainer.setup_training()
    # 保存模型
    filename = time.strftime('./checkpoint/%m%d_%H-%M-%S.pth')
    torch.save(my_trainer.model, filename)

    # Store the resutls in the my_experiment object; this object should contain all the information required to reproduce the results.
    x.append(class_group + args.step_size)
    # 将结果存储在实验对象中；此对象应包含重现结果所需的所有信息。
