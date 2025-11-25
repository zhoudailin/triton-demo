import math
import os
from typing import List

import kaldi_native_fbank as knf
import numpy as np
import soundfile


def _calc_offset_ms(frame_length_ms: float, frame_shift_ms: float) -> float:
    """复制 Triton 的 offset 计算，保证 chunk 之间保留约 20ms 的语音上下文。"""
    offset_ms = 0
    while offset_ms + frame_shift_ms < frame_length_ms:
        offset_ms += frame_shift_ms
    return offset_ms


def _extract_fbank_frames(waveform: np.ndarray, opts: knf.FbankOptions) -> np.ndarray:
    """使用 kaldi_native_fbank 计算整段波形的 fbank。"""
    extractor = knf.OnlineFbank(opts)
    extractor.accept_waveform(opts.frame_opts.samp_freq, waveform)
    num_frames = extractor.num_frames_ready
    if num_frames == 0:
        return np.zeros((0, opts.mel_opts.num_bins), dtype=np.float32)
    frames = np.stack([extractor.get_frame(i) for i in range(num_frames)], axis=0)
    return frames.astype(np.float32)


class CpuStreamingFbank:
    """纯 numpy/kaldi_native_fbank 的流式前端，复刻 Triton 的缓存逻辑（不含 LFR）。"""

    def __init__(self, opts: knf.FbankOptions, chunk_stride: int):
        self.opts = opts
        self.sample_rate = opts.frame_opts.samp_freq
        self.chunk_stride = chunk_stride
        self.scale = float(1 << 15)
        self.frame_stride = int(
            (chunk_stride / self.sample_rate * 1000) // opts.frame_opts.frame_shift_ms
        )
        self.window_size = self.frame_stride + 1
        offset_ms = _calc_offset_ms(opts.frame_opts.frame_length_ms, opts.frame_opts.frame_shift_ms)
        self.offset_samples = int(offset_ms / 1000 * self.sample_rate)
        self.audio_cache = np.zeros(0, dtype=np.float32)
        self.frame_cache = np.zeros((0, opts.mel_opts.num_bins), dtype=np.float32)

    def process_chunk(self, chunk: np.ndarray) -> List[np.ndarray]:
        """处理单个 chunk，返回若干个 61x80 的窗口。"""
        chunk = self._pad_chunk(chunk.astype(np.float32))
        seg = np.concatenate([self.audio_cache, chunk]) if self.audio_cache.size else chunk
        self.audio_cache = (
            seg[-self.offset_samples :].copy() if self.offset_samples > 0 and seg.size else np.zeros(0, dtype=np.float32)
        )
        if seg.size == 0:
            return []
        frames = _extract_fbank_frames(seg * self.scale, self.opts)
        if frames.size == 0:
            return []
        if self.frame_cache.size == 0:
            self.frame_cache = frames
        else:
            self.frame_cache = np.concatenate([self.frame_cache, frames], axis=0)
        return self._collect_windows()

    def flush(self) -> List[np.ndarray]:
        """取出剩余的完整窗口。"""
        return self._collect_windows()

    def _collect_windows(self) -> List[np.ndarray]:
        windows = []
        while self.frame_cache.shape[0] >= self.window_size:
            windows.append(self.frame_cache[: self.window_size].copy())
            self.frame_cache = self.frame_cache[self.frame_stride :]
        return windows

    def _pad_chunk(self, chunk: np.ndarray) -> np.ndarray:
        if len(chunk) >= self.chunk_stride:
            return chunk
        padded = np.zeros(self.chunk_stride, dtype=np.float32)
        padded[: len(chunk)] = chunk
        return padded


def sliding_windows(feats: np.ndarray, window_size: int, stride: int) -> List[np.ndarray]:
    """根据整段特征切出窗口，作为 funASR/funasr 前端的参考。"""
    windows = []
    start = 0
    while start + window_size <= feats.shape[0]:
        windows.append(feats[start : start + window_size].copy())
        start += stride
    return windows


def main():
    wav_file = os.path.join("data", "asr_example.wav")
    speech, sample_rate = soundfile.read(wav_file)
    speech = speech.astype(np.float32)

    chunk_size_s = 0.6
    chunk_stride = int(chunk_size_s * sample_rate)

    opts = knf.FbankOptions()
    opts.frame_opts.samp_freq = sample_rate
    opts.frame_opts.frame_shift_ms = 10.0
    opts.frame_opts.frame_length_ms = 25.0
    opts.frame_opts.window_type = "hamming"
    opts.frame_opts.dither = 0.0
    opts.mel_opts.num_bins = 80

    extractor = CpuStreamingFbank(opts, chunk_stride)

    total_samples = int(math.ceil(len(speech) / chunk_stride) * chunk_stride)
    padded_wave = np.zeros(total_samples, dtype=np.float32)
    padded_wave[: len(speech)] = speech

    streaming_windows = []
    for start in range(0, total_samples, chunk_stride):
        chunk = padded_wave[start : start + chunk_stride]
        streaming_windows.extend(extractor.process_chunk(chunk))
    streaming_windows.extend(extractor.flush())

    reference_feats = _extract_fbank_frames(padded_wave * (1 << 15), opts)
    reference_windows = sliding_windows(reference_feats, extractor.window_size, extractor.frame_stride)

    assert len(streaming_windows) == len(reference_windows), (
        f"stream windows={len(streaming_windows)}, ref windows={len(reference_windows)}"
    )

    diffs = [
        float(np.max(np.abs(a - b)))
        for a, b in zip(streaming_windows, reference_windows)
    ]
    print(f"chunk windows: {len(streaming_windows)}  window_shape={streaming_windows[0].shape}")
    print(f"max abs diff vs funasr reference: {max(diffs):.6f}")
    print(f"mean abs diff: {np.mean(diffs):.6f}")


if __name__ == "__main__":
    main()
