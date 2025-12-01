import json
import math
import os, sys, io
import kaldifeat
import torch

from typing import Dict, Literal
from lru_dict import LRUDict
from feat import Feat
from torch.utils.dlpack import from_dlpack

import triton_python_backend_utils as pb_utils

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", line_buffering=True)


class TritonPythonModel:
    model_config: Dict
    device: Literal['cpu', 'cuda']
    chunk_size: int
    parameters: Dict
    window_size: int
    config_path: str
    config: dict
    opts: kaldifeat.FbankOptions
    fbank: kaldifeat.Fbank
    frame_stride: int
    offset_ms: float
    offset: int
    sample_rate: int
    queue_states: LRUDict

    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])

        if "GPU" in self.model_config["instance_group"][0]["kind"]:
            self.device = "cuda"
        else:
            self.device = "cpu"

        output_speech_config = pb_utils.get_output_config_by_name(self.model_config, "speech")
        self.window_size = output_speech_config["dims"][-2]

        chunk_size_s = 0.6
        self.queue_states = LRUDict(256)

        opts = kaldifeat.FbankOptions()
        opts.frame_opts.dither = 0.0
        opts.frame_opts.samp_freq = 16000
        opts.frame_opts.window_type = "hamming"
        opts.mel_opts.num_bins = 80
        opts.frame_opts.frame_length_ms = 25
        opts.frame_opts.frame_shift_ms = 10

        self.opts = opts
        self.fbank = kaldifeat.Fbank(opts)

        sample_rate = opts.frame_opts.samp_freq
        frame_shift_ms = opts.frame_opts.frame_shift_ms
        frame_length_ms = opts.frame_opts.frame_length_ms

        self.chunk_size = int(chunk_size_s * opts.frame_opts.samp_freq)
        self.frame_stride = int((chunk_size_s * 1000) // frame_shift_ms)
        self.offset_ms = (
                math.ceil((frame_length_ms - frame_shift_ms) / frame_shift_ms)
                * frame_shift_ms
        )
        self.offset = int(self.offset_ms / 1000 * sample_rate)
        self.sample_rate = sample_rate

    def execute(self, requests):
        responses = []
        corrid_index = []
        wave_batch = []
        end_corrid_set = set()
        for request in requests:
            input_wav = pb_utils.get_input_tensor_by_name(request, "wav")
            wav = from_dlpack(input_wav.to_dlpack())[0]
            print(f'wav shape: {wav.shape}')
            wav_len = len(wav)
            if wav_len < self.chunk_size:
                temp = torch.zeros(self.chunk_size, dtype=torch.float32, device=self.device)
                print('chunk_size: ', self.chunk_size)
                temp[0:wav_len] = wav[:]
                wav = temp

            # 状态维护
            start = pb_utils.get_input_tensor_by_name(request, "START").as_numpy()[0][0]
            ready = pb_utils.get_input_tensor_by_name(request, "READY").as_numpy()[0][0]
            corrid = pb_utils.get_input_tensor_by_name(request, "CORRID").as_numpy()[0][
                0
            ]
            end = pb_utils.get_input_tensor_by_name(request, "END").as_numpy()[0][0]

            if start:
                self.queue_states[corrid] = Feat(
                    sample_rate=self.sample_rate,
                    offset=self.offset,
                    frame_stride=self.frame_stride,
                    window_size=self.window_size,
                    device=self.device
                )

            if ready:
                self.queue_states[corrid].add_wavs(wav)

            corrid_index.append(corrid)

            if end:
                end_corrid_set.add(corrid)
            print(self.queue_states)
            wav = self.queue_states[corrid].get_seg_wav()
            wave_batch.append(wav)
            fbank_features = self.fbank(wave_batch)
            print(
                "fbank batch shapes:",
                [tuple(feat.shape) for feat in fbank_features],
            )
            for corrid, frames in zip(corrid_index, fbank_features):
                self.queue_states[corrid].add_frames(frames)
            speech = self.queue_states[corrid].get_frames()
            print(
                f"output speech shape for corrid {corrid}: {tuple(speech.shape)}"
            )

            out_tensor_speech = pb_utils.Tensor("speech", torch.unsqueeze(speech, 0).to("cpu").numpy())
            output_tensors = [out_tensor_speech]
            response = pb_utils.InferenceResponse(output_tensors=output_tensors)
            responses.append(response)

            if corrid in end_corrid_set:
                del self.queue_states[corrid]

        return responses

    def finalize(self):
        print("finalize")
