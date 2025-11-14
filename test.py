import numpy as np
import torch
from kaldifeat import Fbank, FbankOptions
import wave


def read_pcm_wave(file_path):
    with wave.open(file_path, "rb") as wav_file:
        # 获取文件的基本信息
        n_channels = wav_file.getnchannels()  # 声道数
        sample_width = wav_file.getsampwidth()  # 采样宽度（字节数）
        frame_rate = wav_file.getframerate()  # 采样率（帧率）
        n_frames = wav_file.getnframes()  # 总帧数

        # 读取所有音频数据（作为字节流）
        pcm_data = wav_file.readframes(n_frames)

        # 将字节流转换为 numpy 数组
        pcm_array = np.frombuffer(pcm_data, dtype=np.int16)  # 根据采样宽度调整数据类型

        # 如果是立体声（2个声道），我们需要对数据进行重新排列
        if n_channels == 2:
            pcm_array = pcm_array.reshape((-1, 2))  # 每一对数值是左右声道的 PCM 数据

        return pcm_array, frame_rate


pcm, frame_rate = read_pcm_wave("data/asr_example.wav")
opts = FbankOptions()
opts.frame_opts.dither = 0.0
opts.frame_opts.window_type = "hamming"
opts.mel_opts.num_bins = 80
opts.frame_opts.frame_shift_ms = 10
opts.frame_opts.frame_length_ms = 25
opts.frame_opts.samp_freq = 16000
opts.device = torch.device("cuda")
f = Fbank(opts)
fbank = f(torch.tensor(pcm, dtype=torch.float32).to("cuda"))
print(pcm.shape, fbank, fbank.shape)
