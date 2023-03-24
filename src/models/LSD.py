# pylint: disable-all
import torch
from torch import nn


class LSD(nn.Module):
    """A simple linear neural network for fast development"""

    def __init__(self):
        super(LSD, self).__init__()
        # convolutional layer
        self.conv1 = nn.Conv2d(3, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # convolutional layer
        x = self.conv1(x)
        # output        
        return torch.sigmoid(x)
    
    def predict(self, x):
        activations = self.forward(x)
        return (activations > .5).int()