import torch

def adam(parameters, lr=0.001, betas=(0.9, 0.999), weight_decay=0):
    return torch.optim.Adam(parameters, lr=lr, betas=betas, weight_decay=weight_decay)

def sgd(parameters, lr=0.1, momentum=0.9, weight_decay=0):
    return torch.optim.SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)