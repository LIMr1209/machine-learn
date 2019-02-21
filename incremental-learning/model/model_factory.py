''' Incremental-Classifier Learning 
 Authors : Khurram Javed, Muhammad Talha Paracha
 Maintainer : Khurram Javed
 Lab : TUKL-SEECS R&D Lab
 Email : 14besekjaved@seecs.edu.pk '''

import model.resnet32 as res
from model.resnet152 import ResNet152

class ModelFactory():
    def __init__(self):
        pass

    @staticmethod
    def get_model(model_type, dataset="CIFAR100"):

        if model_type == "resnet32":
            if dataset == "MNIST":
                return res.resnet32mnist(10)
            elif dataset == "CIFAR10":
                return res.resnet32(10)
            return res.resnet32(100)

        elif model_type == 'resnet152':
            return ResNet152()



        else:
            print("Unsupported model; either implement the model in model/ModelFactory or choose a different model")
            assert (False)


