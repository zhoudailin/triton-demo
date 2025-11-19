import json
from typing import Dict, List
import os, sys, io
import triton_python_backend_utils as pb_utils
import torch
import yaml
from torch import from_dlpack
from lru_dict import LRUDict
from feat import Feat
import kaldifeat

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", line_buffering=True)


class TritonPythonModel:
    model_config: Dict
    chunk_size: float
    parameters: Dict
    chunk_size_s: float
    config_path: str
    config: dict
    seq_feat: LRUDict
    opts: kaldifeat.FbankOptions
    feature_extractor: kaldifeat.Fbank
    frame_stride: float
    offset_ms: float
    sample_rate: float
    device: str

    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])
        output0_config = pb_utils.get_output_config_by_name(self.model_config, "speech")

        self.parameters = {}
        for param in self.model_config["parameters"].items():
            key, value = param
            value = value["string_value"]
            self.parameters[key] = value
        self.chunk_size_s = float(self.parameters["chunk_size_s"])
        self.config_path = self.parameters["config_path"]
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        self.feature_size = output0_config["dims"][-1]
        self.decoding_window = output0_config["dims"][-2]
        opts = kaldifeat.FbankOptions()
        opts.frame_opts.dither = 0.0  # 随机噪声
        opts.frame_opts.window_type = self.config["frontend_conf"]["window"]  # 窗函数
        opts.mel_opts.num_bins = int(self.config["frontend_conf"]["n_mels"])  # 梅尔频带数
        opts.frame_opts.frame_shift_ms = float(self.config["frontend_conf"]["frame_shift"])  # 帧位移
        opts.frame_opts.frame_length_ms = float(self.config["frontend_conf"]["frame_length"])  # 帧长度
        opts.frame_opts.samp_freq = int(self.config["frontend_conf"]["fs"])  # 帧率
        opts.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.opts = opts
        self.feature_extractor = kaldifeat.Fbank(self.opts)

        self.seq_feat = LRUDict(1024)

        sample_rate = opts.frame_opts.samp_freq
        frame_shift_ms = opts.frame_opts.frame_shift_ms
        frame_length_ms = opts.frame_opts.frame_length_ms

        print(f'-------------\n{self.chunk_size_s} {sample_rate}')
        print(f'-------------\n{type(self.chunk_size_s)} {type(sample_rate)}')
        self.chunk_size = self.chunk_size_s * sample_rate
        self.frame_stride = (self.chunk_size_s * 1000) // frame_shift_ms
        self.offset_ms = self.get_offset(frame_length_ms, frame_shift_ms)
        self.sample_rate = sample_rate

    @staticmethod
    def get_offset(frame_length_ms, frame_shift_ms):
        offset_ms = 0
        while offset_ms + frame_shift_ms < frame_length_ms:
            offset_ms += frame_shift_ms
        return offset_ms

    def execute(self, requests):
        responses = []
        total_waves = []
        batch_seqid = []
        end_seqid = {}
        for request in requests:
            input0 = pb_utils.get_input_tensor_by_name(request, "WAV")
            print(input0, type(input0))
            wav = from_dlpack(input0.to_dlpack())[0]
            print(wav, type(wav), wav.shape)
            wav_len = len(wav)
            if wav_len < self.chunk_size:
                temp = torch.zeros(
                    self.chunk_size, dtype=torch.float32, device=self.device
                )
                temp[0:wav_len] = wav[:]
                wav = temp

            in_start = pb_utils.get_input_tensor_by_name(request, "START")
            start = in_start.as_numpy()[0][0]
            in_ready = pb_utils.get_input_tensor_by_name(request, "READY")
            ready = in_ready.as_numpy()[0][0]
            in_corrid = pb_utils.get_input_tensor_by_name(request, "CORRID")
            corrid = in_corrid.as_numpy()[0][0]
            in_end = pb_utils.get_input_tensor_by_name(request, "END")
            end = in_end.as_numpy()[0][0]

            if start:
                self.seq_feat[corrid] = Feat(
                    corrid,
                    self.offset_ms,
                    self.sample_rate,
                    self.frame_stride,
                    self.device,
                )
            if ready:
                self.seq_feat[corrid].add_audio(wav)
            batch_seqid.append(corrid)
            if end:
                end_seqid[corrid] = 1
            wav = self.seq_feat[corrid].get_seg_wav() * 32768
            total_waves.append(wav)
        features = self.feature_extractor(total_waves)
        for corrid, frames in zip(batch_seqid, features):
            self.seq_feat[corrid].add_frames(frames)
            speech = self.seq_feat[corrid].get_frames(self.decoding_window)
            out_tensor0 = pb_utils.Tensor("speech", torch.unsqueeze(speech, 0).to("cpu").numpy())
            output_tensors = [out_tensor0]
            response = pb_utils.InferenceResponse(output_tensors=output_tensors)
            responses.append(response)
            if corrid in end_seqid:
                del self.seq_feat[corrid]
        return responses

    def finalize(self):
        print("finalize")
