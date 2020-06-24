import torch
import torch.nn as nn

class ConvHeadEvalModel(nn.Module):
    """A thin wrapper to average the output of a conv-headed model
    across all spatial dimensions
    i.e. a conv-headed model's output has shape [batch_size, num_classes, h, w]
    where h and w are feature map spatial dimension
    This wrapper average the output across the last two dimensions to get shape
    [batch_size, num_classes]
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        x = self.model(x)
        x = torch.mean(x, dim = (2, 3))
        return x