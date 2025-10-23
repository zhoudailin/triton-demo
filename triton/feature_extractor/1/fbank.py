from typing import List

import kaldifeat
import torch


class Fbank(torch.nn.Module):
    """
    包装为pytorch模块
    """
    def __init__(self, opts):
        super(Fbank, self).__init__()
        self.fbank = kaldifeat.Fbank(opts)

    def forward(self, waves: List[torch.Tensor]):
        return self.fbank(waves)
