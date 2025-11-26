import numpy as np


class FeatStream:
    def __init__(self,
                 corrid,
                 sample_rate,
                 offset_ms,
                 frame_stride,
                 lfr_m=7,
    ):

        self.corrid = corrid
        self.sample_rate = sample_rate
        self.offset_ms = offset_ms
        self.frame_stride = frame_stride

        self.wav_buf = np.array([], dtype=np.float32)

        # fbank frame buffer (float32, shape [T, 80])
        self.frame_buf = None

        # LFR m
        self.lfr_m = lfr_m

    # add new wav
    def add_wavs(self, wav):
        self.wav_buf = np.concatenate([self.wav_buf, wav], axis=0)

    def pop_chunk(self, chunk_size_samples):
        if len(self.wav_buf) < chunk_size_samples:
            padded = np.zeros(chunk_size_samples, dtype=np.float32)
            padded[:len(self.wav_buf)] = self.wav_buf
            chunk = padded
            self.wav_buf = np.zeros(self.offset_ms * self.sample_rate // 1000, dtype=np.float32)
            return chunk
        else:
            chunk = self.wav_buf[:chunk_size_samples]
            keep_len = int(self.offset_ms * self.sample_rate // 1000)
            self.wav_buf = self.wav_buf[-keep_len:]
            return chunk

    # append new frames
    def add_frames(self, frames):
        # frames: [num_frames, 80]
        if self.frame_buf is None:
            # first chunk → LFR pad 3 frames at start
            pad = np.tile(frames[0], ((self.lfr_m - 1) // 2, 1))
            self.frame_buf = np.concatenate([pad, frames], axis=0)
        else:
            self.frame_buf = np.concatenate([self.frame_buf, frames], axis=0)

    # return fixed window (61 frames)
    def pop_window(self, win_size):
        if self.frame_buf is None or len(self.frame_buf) < win_size:
            return None

        seg = self.frame_buf[:win_size]
        self.frame_buf = self.frame_buf[self.frame_stride:]
        return seg
