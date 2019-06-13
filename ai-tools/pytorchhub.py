import torch
model_list = torch.hub.list('pytorch/vision')
model = torch.hub.load('pytorch/vision', 'alexnet', pretrained=True)
model.eval()
