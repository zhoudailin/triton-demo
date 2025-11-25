from typing import List

import kaldi_native_fbank as knf
import torch


class Fbank(torch.nn.Module):
    """
    包装为pytorch模块
    """

    def __init__(self, opts):
        super(Fbank, self).__init__()
        self.fbank = knf.OnlineFbank(opts)

    def forward(self, waves: List[torch.Tensor]):
        return self.fbank.accept_waveform(16000,waves)
