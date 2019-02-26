''' Incremental-Classifier Learning 
 Authors : Khurram Javed, Muhammad Talha Paracha
 Maintainer : Khurram Javed
 Lab : TUKL-SEECS R&D Lab
 Email : 14besekjaved@seecs.edu.pk '''

import logging

import numpy as np
import torch
import torch.nn.functional as F
from torchnet.meter import confusionmeter

logger = logging.getLogger('iCARL')


class EvaluatorFactory():
    '''
    This class is used to get different versions of evaluators
    '''

    def __init__(self):
        pass

    @staticmethod
    def get_evaluator(testType="nmc", cuda=True):
        if testType == "nmc":
            return NearestMeanEvaluator(cuda)
        if testType == "trainedClassifier":
            return softmax_evaluator(cuda)


class NearestMeanEvaluator():
    '''
    Nearest Class Mean based classifier. Mean embedding is computed and stored; at classification time, the embedding closest to the 
    input embedding corresponds to the predicted class.
    最近的基于类均值的分类器。计算并存储平均嵌入；在分类时，嵌入最接近输入嵌入对应于预测类。
    '''

    def __init__(self, cuda):
        self.cuda = cuda
        self.means = None
        self.totalFeatures = np.zeros((100, 1))  # 100 是类数目?  9?1000?

    def evaluate(self, model, loader, step_size=10, kMean=False):
        '''
        :param model: Train model
        :param loader: Data loader
        :param step_size: Step size for incremental learning
        :param kMean: Doesn't work very well so don't use; Will be removed in future versions 
        :return: 
        '''
        model.eval()
        if self.means is None:
            self.means = np.zeros((100, model.featureSize))  # 100 是类数目?  9?1000?
        correct = 0

        for data, y, target in loader:
            if self.cuda:
                data, target = data.to(torch.device('cuda')), target.to(torch.device('cuda'))
                self.means = self.means.cuda()
            output = model(data, True).unsqueeze(1)
            result = (output.data - self.means.float())

            result = torch.norm(result, 2, 2)
            if kMean:
                result = result.cpu().numpy()
                tempClassifier = np.zeros((len(result), int(100 / step_size)))  # 100 是类数目?  9?1000?
                for outer in range(0, len(result)):
                    for tempCounter in range(0, int(100 / step_size)):   # 100 是类数目?  9?1000?
                        tempClassifier[outer, tempCounter] = np.sum(
                            result[tempCounter * step_size:(tempCounter * step_size) + step_size])
                for outer in range(0, len(result)):
                    minClass = np.argmin(tempClassifier[outer, :])
                    result[outer, 0:minClass * step_size] += 300000
                    result[outer, minClass * step_size:(minClass + 1) * step_size] += 300000
                result = torch.from_numpy(result)
                if self.cuda:
                    result = result.to(torch.device('cuda'))
            _, pred = torch.min(result, 1)
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        return 100. * correct / len(loader.dataset)

    def get_confusion_matrix(self, model, loader, size):
        '''
        
        :param model: Trained model
        :param loader: Data iterator
        :param size: Size of confusion matrix (Equal to largest possible label predicted by the model)  混淆矩阵的大小（等于模型预测的所有可能的标签）
        :return: 
        '''
        model.eval()
        test_loss = 0
        correct = 0
        # Get the confusion matrix object
        cMatrix = confusionmeter.ConfusionMeter(size, True)

        for data, y, target in loader:
            if self.cuda:
                data, target = data.cuda(), target.to(torch.device('cuda'))
                self.means = self.means.cuda()
            output = model(data, True).unsqueeze(1)
            result = (output.data - self.means.float())

            result = torch.norm(result, 2, 2)
            # NMC for classification
            _, pred = torch.min(result, 1)
            # Evaluate results
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            # Add the results in appropriate places in the matrix.
            cMatrix.add(pred, target.data.view_as(pred))

        test_loss /= len(loader.dataset)
        # Get 2d numpy matrix to remove the dependency of other code on confusionmeter
        img = cMatrix.value()
        return img

    def update_means(self, model, train_loader, classes=100):
        '''
        This method updates the mean embedding using the train data; DO NOT pass test data iterator to this. 
        :param model: Trained model
        :param train_loader: data iterator
        :param classes: Total number of classes
        :return: 
        '''
        # Set the mean to zero
        if self.means is None:
            self.means = np.zeros((classes, model.featureSize))
        self.means *= 0
        self.classes = classes
        self.means = np.zeros((classes, model.featureSize))
        self.totalFeatures = np.zeros((classes, 1)) + .001
        logger.debug("Computing means")
        # Iterate over all train Dataset
        for batch_id, (data, y, target) in enumerate(train_loader):
            # Get features for a minibactch
            if self.cuda:
                data = data.to(torch.device('cuda'))
            features = model.forward(data, True)
            # Convert result to a numpy array
            featuresNp = features.data.cpu().numpy()
            # Accumulate the results in the means array
            # print (self.means.shape,featuresNp.shape)
            np.add.at(self.means, target, featuresNp)
            # Keep track of how many instances of a class have been seen. This should be an array with all elements = classSize
            np.add.at(self.totalFeatures, target, 1)

        # Divide the means array with total number of instan    ces to get the average
        # print ("Total instances", self.totalFeatures)
        self.means = self.means / self.totalFeatures
        self.means = torch.from_numpy(self.means)
        # Normalize the mean vector
        self.means = self.means / torch.norm(self.means, 2, 1).unsqueeze(1)
        self.means[self.means != self.means] = 0
        self.means = self.means.unsqueeze(0)

        logger.debug("Mean vectors computed")
        # Return
        return


class softmax_evaluator():
    '''
    Evaluator class for softmax classification  softmax 分类器
    '''

    def __init__(self, cuda):
        self.cuda = cuda
        self.means = None
        self.totalFeatures = np.zeros((100, 1))    # 100 是类数目?  9?1000?

    def evaluate(self, model, loader, scale=None, thres=False, older_classes=None, step_size=10, descriptor=False,
                 falseDec=False):
        '''
        :param model: Trained model
        :param loader: Data iterator
        :param scale: Scale vector computed by dynamic threshold moving
        :param thres: If true, use scaling
        :param older_classes: Will be removed in next iteration
        :param step_size: Step size for incremental learning
        :param descriptor: Will be removed in next iteration; used to compare the results with a recent paper by Facebook. 
        :param falseDec: Will be removed in the next iteration.
        :return:
        '''
        model.eval()
        correct = 0
        if scale is not None:
            scale = np.copy(scale)
            scale = scale / np.max(scale)
            # print ("Gets here")
            scaleTemp = np.copy(scale)
            if thres:
                for x in range(0, len(scale)):
                    temp = 0
                    for y in range(0, len(scale)):
                        if x == y:
                            pass
                        else:
                            temp = temp + (scale[y] / scale[x])
                        scaleTemp[x] = temp
                scale = scaleTemp
            else:
                scale = 1 / scale

            scale = scale / np.linalg.norm(scale, 1)
            scale = torch.from_numpy(scale).unsqueeze(0)
            if self.cuda:
                scale = scale.to(torch.device('cuda'))
        tempCounter = 0
        for data, y, target in loader:
            if self.cuda:
                data, target = data.to(torch.device('cuda')), target.to(torch.device('cuda'))
            if thres:
                output = model(data)
                output = output * scale.float()
            elif scale is not None:
                # print("Gets here, getting outputs")
                output = model(data, scale=scale.float())
            else:
                output = model(data)
            if descriptor:
                # To compare with FB paper
                # output = output/scale.float()
                outputTemp = output.data.cpu().numpy()
                targetTemp = target.data.cpu().numpy()
                if falseDec:
                    for a in range(0, len(targetTemp)):
                        random = np.random.choice(len(older_classes) + step_size, step_size, replace=False).tolist()
                        if targetTemp[a] in random:
                            pass
                        else:
                            random[0] = targetTemp[a]
                        for b in random:
                            outputTemp[a, b] += 20
                else:
                    for a in range(0, len(targetTemp)):
                        outputTemp[a, int(float(targetTemp[a]) / step_size) * step_size:(int(
                            float(targetTemp[a]) / step_size) * step_size) + step_size] += 20
                if tempCounter == 0:
                    print(int(float(targetTemp[a]) / step_size) * step_size,
                          (int(float(targetTemp[a]) / step_size) * step_size) + step_size)
                    tempCounter += 1

                output = torch.from_numpy(outputTemp)
                if self.cuda:
                    output = output.to(torch.device('cuda'))
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        return 100. * correct / len(loader.dataset)

    def get_confusion_matrix(self, model, loader, size, scale=None, older_classes=None, step_size=10, descriptor=False):
        '''
        :return: Returns the confusion matrix on the data given by loader
        '''
        model.eval()
        test_loss = 0
        correct = 0
        # Initialize confusion matrix
        cMatrix = confusionmeter.ConfusionMeter(size, True)

        # Checks is threshold moving should be used
        if scale is not None:
            scale = np.copy(scale)
            scale = scale / np.max(scale)
            scale = 1 / scale
            scale = torch.from_numpy(scale).unsqueeze(0)
            if self.cuda:
                scale = scale.to(torch.device('cuda'))

        # Iterate over the data and stores the results in the confusion matrix
        for data, y, target in loader:
            if self.cuda:
                data, target = data.to(torch.device('cuda')), target.to(torch.device('cuda'))
            if scale is not None:
                output = model(data, scale=scale.float())
            else:
                output = model(data)

            if descriptor:
                # To compare with FB paper
                outputTemp = output.data.cpu().numpy()
                targetTemp = target.data.cpu().numpy()
                for a in range(0, len(targetTemp)):
                    outputTemp[a, int(float(targetTemp[a]) / step_size) * step_size:(int(
                        float(targetTemp[a]) / step_size) * step_size) + step_size] += 20
                output = torch.from_numpy(outputTemp)
                if self.cuda:
                    output = output.to(torch.device('cuda'))
                output = output

            # c = F.nll_loss(output, target, size_average=False).data  tensor(97.6165, device='cuda:0')
            # test_loss += F.nll_loss(output, target, size_average=False).item()  # 即将弃用
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            cMatrix.add(pred.squeeze(), target.data.view_as(pred).squeeze())

        # Returns normalized matrix.
        test_loss /= len(loader.dataset)
        img = cMatrix.value()
        return img


if __name__ == '__main__':
    torch.manual_seed(100)
    the_model = torch.load('../checkpoint/0220_12-59-07.pth')
    the_model.to(torch.device('cuda'))
    # print(the_model.parameters)
    the_model.eval()
    softmax = softmax_evaluator(True)
    input = torch.randn(1, 3, 32, 32).cuda()
    print(input)
    data = input
    output = the_model(data)
    pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
    print(pred)
