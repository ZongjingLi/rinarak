import torch
import torch.nn as nn

class SetNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
    
    def forward(self, x):
        return x