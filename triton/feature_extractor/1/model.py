import json
from typing import Dict, List
import os, sys, io
import triton_python_backend_utils as pb_utils
import torch
import yaml
from torch import from_dlpack
from lru_dict import LRUDict
from feat import Feat

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", line_buffering=True)


class TritonPythonModel:
    model_config: Dict
    chunk_size: int
    parameters: Dict
    chunk_size_s: str
    config_path: str
    config: dict
    seq_feat: LRUDict

    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])

        self.parameters = {}
        for param in self.model_config["parameters"].items():
            key, value = param
            value = value["string_value"]
            self.parameters[key] = value
        self.chunk_size_s = self.parameters["chunk_size_s"]
        self.config_path = self.parameters["config_path"]

        self.seq_feat = LRUDict(1024)

        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

        chunk_size_s = float(self.parameters["chunk_size_s"])
        sample_rate = self.config["frontend_conf"]["fs"]
        self.chunk_size = int(chunk_size_s * sample_rate)
        print(chunk_size_s, sample_rate, self.chunk_size)

    def execute(self, requests):
        responses = []
        for request in requests:
            input0 = pb_utils.get_input_tensor_by_name(request, "wav")
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

        response = pb_utils.InferenceResponse(output_tensors=torch.tensor([1, 2, 3]))
        return [response]

    def finalize(self):
        print("finalize")
