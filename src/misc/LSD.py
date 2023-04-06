import torch
from torch import nn


class LSD(nn.Module):
    """A simple linear neural network for fast development"""

    def __init__(self):
        super(LSD, self).__init__()
        # convolutional layer
        self.conv1 = nn.Conv2d(3, 2, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # convolutional layer
        x = self.conv1(x)
        # output
        preds = torch.sigmoid(x)
        preds = preds.flatten(2, -1)  # Shape (B, C, H*W)
        preds = preds.permute(0, 2, 1)  # Shape (B, H*W, C)
        return preds

    def predict(self, x):
        activations = self.forward(x)
        return (activations > 0.5).int()
