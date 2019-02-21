import numpy as np
import torch as t
import data_handler

from trainer.evaluator import softmax_evaluator

t.manual_seed(100)
the_model = t.load('./checkpoint/0220_12-59-07.pth')
the_model.to(t.device('cuda'))
# print(the_model.parameters)
the_model.eval()
softmax = softmax_evaluator(True)
dataset = data_handler.DatasetFactory.get_dataset('CIFAR100')
test_dataset_loader = data_handler.IncrementalLoader(dataset.test_data.test_data,
                                                     dataset.test_data.test_labels,
                                                     dataset.labels_per_class_test, dataset.classes,
                                                     [1, 2], transform=dataset.test_transform,
                                                     cuda=True, oversampling=False
                                                     )

print(input)
output = the_model(input)
pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
print(pred)
