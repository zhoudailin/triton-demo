from typing import List

import torch
import kaldifeat


class FbankModel(torch.nn.Module):
    def __init__(self, opts):
        super(kaldifeat.Fbank, self).__init__()
        self.fbank = kaldifeat.Fbank(opts)

    def forward(self, waves: List[torch.Tensor]):
        return self.fbank(waves)
