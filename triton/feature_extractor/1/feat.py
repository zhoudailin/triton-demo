import torch


class Feat(object):
    def __init__(self, seqid, offset_ms, sample_rate, frame_stride, device="cpu"):
        self.seqid = seqid
        self.sample_rate = sample_rate
        self.wav: torch.Tensor = torch.tensor([], device=device)
        self.offset = int(offset_ms / 1000 * sample_rate)
        self.frames = None
        self.frame_stride = int(frame_stride)
        self.device = device
        self.lfr_m = 7

    def add_wavs(self, wav: torch.Tensor):
        wav = wav.to(self.device)
        self.wav = torch.cat((self.wav, wav), axis=0)

    def get_seg_wav(self):
        seg = self.wav[:]
        self.wav = self.wav[-self.offset :]
        return seg

    def add_frames(self, frames: torch.Tensor):
        """
        frames: seq_len x feat_sz
        """
        if self.frames is None:
            self.frames = torch.cat(
                (frames[0, :].repeat((self.lfr_m - 1) // 2, 1), frames), axis=0
            )
        else:
            self.frames = torch.cat([self.frames, frames], axis=0)

    def get_frames(self, num_frames: int):
        seg = self.frames[0:num_frames]
        self.frames = self.frames[self.frame_stride :]
        return seg
