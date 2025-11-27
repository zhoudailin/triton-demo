import torch


class Feat(object):
    def __init__(self, sample_rate: int, offset: int, frame_stride: int, window_size: int, device):
        self.sample_rate = sample_rate
        self.offset = offset
        self.frame_stride = frame_stride
        self.window_size = window_size

        self.wav = torch.tensor([], device=device)
        self.frames = None
        self.lfr_m = 7

    def add_wavs(self, wav: torch.Tensor):
        print(self.wav.device, wav.device)
        self.wav = torch.cat((self.wav, wav), axis=0)

    def get_seg_wav(self):
        seg = self.wav[:]
        self.wav = self.wav[-self.offset:]
        return seg * 32768

    def add_frames(self, frames: torch.Tensor):
        if self.frames is None:
            self.frames = torch.cat(
                (frames[0, :].repeat((self.lfr_m - 1) // 2, 1), frames), axis=0
            )
        else:
            self.frames = torch.cat([self.frames, frames], axis=0)

    def get_frames(self):
        seg = self.frames[0:self.window_size]
        self.frames = self.frames[self.frame_stride:]
        return seg
