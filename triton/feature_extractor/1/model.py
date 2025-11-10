import json
from typing import Dict, List
import os, sys, io
import kaldifeat
import triton_python_backend_utils as pb_utils
import torch
import yaml
from torch import from_dlpack
from fbank import Fbank

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

    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])

        self.parameters = {}
        for param in self.model_config["parameters"].items():
            key, value = param
            value = value["string_value"]
            self.parameters[key] = value
        self.chunk_size_s = self.parameters["chunk_size_s"]
        self.config_path = self.parameters["config_path"]

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
        return [1]

    def finalize(self):
        print("finalize")
