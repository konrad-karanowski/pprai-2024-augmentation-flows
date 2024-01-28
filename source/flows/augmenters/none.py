import torch
from torch import nn


class NoneAugmenter(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return x
