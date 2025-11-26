import json
import math
import os, sys, io
import yaml

from typing import Dict
from lru_dict import LRUDict
from featstream import Feat, FeatStream

import numpy as np
import kaldi_native_fbank as knf
import triton_python_backend_utils as pb_utils

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", line_buffering=True)


class TritonPythonModel:
    model_config: Dict
    chunk_size: int
    parameters: Dict
    chunk_size_s: float
    config_path: str
    config: dict
    seq_feat: LRUDict
    opts: knf.FbankOptions
    fbank: knf.OnlineFbank
    frame_stride: float
    offset_ms: float
    sample_rate: float
    seq_states: Dict[str, FeatStream]

    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])
        chunk_size_s = 0.6

        opts = knf.FbankOptions()
        opts.frame_opts.dither = 0.0
        opts.frame_opts.samp_freq = 16000
        opts.frame_opts.window_type = 'hamming'
        opts.mel_opts.num_bins = 80
        opts.frame_opts.frame_length_ms = 25
        opts.frame_opts.frame_shift_ms = 10
        self.opts = opts
        self.fbank = knf.OnlineFbank(opts)

        self.seq_feat = LRUDict(256)

        sample_rate = opts.frame_opts.samp_freq
        frame_shift_ms = opts.frame_opts.frame_shift_ms
        frame_length_ms = opts.frame_opts.frame_length_ms

        self.chunk_size = int(chunk_size_s * opts.frame_opts.samp_freq)
        self.frame_stride = (self.chunk_size_s * 1000) // frame_shift_ms
        self.offset_ms = math.ceil((frame_length_ms - frame_shift_ms) / frame_shift_ms) * frame_shift_ms
        self.sample_rate = sample_rate

    def execute(self, requests):
        responses = []
        for request in requests:
            input_wav = pb_utils.get_input_tensor_by_name(request, "wav")
            print('input_wav: ', input_wav, type(input_wav))
            wav = input_wav.as_numpy().astype(np.float32)
            print(wav, type(wav), wav.shape)
            wav_len = len(wav)
            if wav_len < self.chunk_size:
                temp = np.zeros(self.chunk_size, dtype=np.float32)
                temp[0:wav_len] = wav[:]
                wav = temp

            # 状态维护
            start = pb_utils.get_input_tensor_by_name(request, "START").as_numpy()[0][0]
            ready = pb_utils.get_input_tensor_by_name(request, "READY").as_numpy()[0][0]
            corrid = pb_utils.get_input_tensor_by_name(request, "CORRID").as_numpy()[0][0]
            end = pb_utils.get_input_tensor_by_name(request, "END").as_numpy()[0][0]

            if start:
                self.seq_states[corrid] = FeatStream(
                    corrid=corrid,
                    sample_rate=self.sample_rate,
                    offset_ms=self.offset_ms,
                    frame_stride=self.frame_stride
                )

            stream = self.seq_states[corrid]

            if ready:
                stream.add_wavs(wav)

            chunk = stream.pop_chunk(self.chunk_size_samples)

            fb = knf.OnlineFbank(self.fbank_opts)
            fb.accept_waveform(self.sample_rate, chunk)

            # pull all available frames
            frames = []
            while fb.num_frames_ready > 0:
                f = fb.get_frame(0)
                # move the buffer forward inside OnlineFbank
                fb.pop_frame()
                frames.append(f)

            frames = np.stack(frames, axis=0) if frames else np.zeros((0, self.feature_dim), np.float32)

            # append frames to stream buffer
            if len(frames) > 0:
                stream.add_frames(frames)

            # try pop one fixed window
            seg = stream.pop_window(self.win_size)

            if seg is None:
                seg = np.zeros((self.win_size, self.feature_dim), np.float32)

            out = pb_utils.Tensor("speech", seg.reshape(1, self.win_size, self.feature_dim))
            responses.append(pb_utils.InferenceResponse(output_tensors=[out]))

            # cleanup
            if end:
                del self.seq_states[corrid]
        return responses

    def finalize(self):
        print("finalize")
