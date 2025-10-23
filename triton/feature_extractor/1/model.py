import json
import kaldifeat
import triton_python_backend_utils as pb_utils
import torch
import yaml
from fbank import Fbank


class TritonPythonModel:
    def __init__(self):
        self.feature_extractor = None
        self.opts = None
        self.output_dtype = None
        self.device = None
        self.model_config = None

    def initialize(self, args):
        self.model_config = model_config = json.loads(args["model_config"])

        if args['model_instance_kind'] == 'GPU':
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        output_config = pb_utils.get_output_config_by_name(model_config, 'speech')
        self.output_dtype = pb_utils.triton_string_to_numpy(output_config['data_type'])

        parameters = self.model_config['parameters']
        config_path = parameters['config_path']['string_value']
        chunk_size_s = parameters['chunk_size_s']['string_value']

        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        opts = kaldifeat.FbankOptions()
        opts.frame_opts.dither = 0.0  # 随机噪声
        opts.frame_opts.window_type = config["frontend_conf"]["window"]  # 窗函数
        opts.mel_opts.num_bins = int(config["frontend_conf"]["n_mels"])  # mel滤波器数量，决定了FBank特征维度，对于深度学习一般用80
        opts.frame_opts.frame_shift_ms = float(config["frontend_conf"]["frame_shift"])  # 帧之间overlap
        opts.frame_opts.frame_length_ms = float(config["frontend_conf"]["frame_length"])  # 帧长，决定了每个帧的持续时间，一般用25ms
        opts.frame_opts.samp_freq = int(config["frontend_conf"]["fs"])  # 采样率，一般用16000Hz
        opts.device = torch.device(self.device)
        self.opts = opts
        self.feature_extractor = Fbank(self.opts)

    def execute(self, requests):
        pass

    def finalize(self):
        pass


if __name__ == '__main__':
    model = TritonPythonModel()
