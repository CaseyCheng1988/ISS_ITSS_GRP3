import numpy as np
from torch import nn

# custom normalized sigmoid function
def sigmoid(A):
    ret = 1/(1+np.exp(-A))
    ret /= np.sum(ret)
    return ret

# allow activation function customization
def activation_func(activation):
    return nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
        ['selu', nn.SELU(inplace=True)],
        ['none', nn.Identity()]
    ])[activation]

# fully-connected layer adjustable according to input dimension and multiplier factor
def fc_function(in_channels, multiplier):
    out_channels = int(multiplier * in_channels)
    return nn.Linear(in_channels, out_channels, bias=True), out_channels