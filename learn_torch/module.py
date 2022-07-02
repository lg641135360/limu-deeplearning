import torch
from torch import nn


class HN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        output = input + 1
        return output


hn = HN()
x = torch.tensor(1.0)
output = hn(x)
print(output)