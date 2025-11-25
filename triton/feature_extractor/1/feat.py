import numpy as np


class Feat(object):
    def __init__(self, corrid, offset_ms, sample_rate, frame_stride):
        self.corrid = corrid
        self.sample_rate = sample_rate
        self.audios: np.ndarray = np.array([])
        self.offset = int(offset_ms / 1000 * sample_rate)
        self.frames = None
        self.frame_stride = int(frame_stride)
        self.lfr_m = 7

    def add_audio(self, audio: np.ndarray):
        print('add_audio: ', self.audios, audio)
        self.audios = np.concatenate([self.audios, audio])

    def get_seg_wav(self):
        seg = self.audios[:]
        self.audios = self.audios[-self.offset:]
        return seg

    def add_frames(self, frames: np.ndarray):
        if self.frames is None:
            self.frames = np.concatenate(
                (frames[0, :].repeat((self.lfr_m - 1) // 2, 1), frames)
            )
        else:
            self.frames = np.concatenate([self.frames, frames])

    def get_frames(self, num_frames: int):
        seg = self.frames[0:num_frames]
        self.frames = self.frames[self.frame_stride:]
        return seg
