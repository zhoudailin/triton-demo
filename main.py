import math

import numpy as np
from tritonclient.grpc import InferenceServerClient, InferInput, InferRequestedOutput

from funasr import AutoModel

chunk_size = [0, 10, 5]  # [0, 10, 5] 600ms, [0, 8, 4] 480ms
encoder_chunk_look_back = 5  # number of chunks to lookback for encoder self-attention
decoder_chunk_look_back = 2  # number of encoder chunks to lookback for decoder cross-attention

# model = AutoModel(model="paraformer-zh-streaming", model_revision="v2.0.4")

import soundfile
import os

client = InferenceServerClient(url="localhost:8001")

wav_file = os.path.join("data/asr_example.wav")
speech, sample_rate = soundfile.read(wav_file)
chunk_stride = chunk_size[1] * 960  # 600ms

cache = {}
total_chunk_num = math.ceil(len(speech) / chunk_stride)
print(f'total_chunk_num: {total_chunk_num}')
for i in range(total_chunk_num):
    # speech_chunk = speech[i * chunk_stride:(i + 1) * chunk_stride]
    # is_final = i == total_chunk_num - 1
    # wav_np = speech_chunk.astype(np.float32).reshape(1, -1)
    # inputs = [
    #     InferInput('wav', wav_np.shape, 'FP32'),
    # ]
    # inputs[0].set_data_from_numpy(wav_np)
    # outputs = [InferRequestedOutput('transcripts')]
    enc = np.random.randn(10, 512).astype(np.float32).reshape(1, 10, 512)  # [10, 512]
    enc_len = np.array([10], dtype=np.int32).reshape(1, -1)
    acoustic_embeds = np.random.randn(1, 512).astype(np.float32).reshape(1, 1, 512)  # [1, 512]
    acoustic_embeds_len = np.array([1], dtype=np.int32).reshape(1, -1)

    inputs = [
        InferInput("enc", enc.shape, 'FP32'),
        InferInput("enc_len", enc_len.shape, 'INT32'),
        InferInput("acoustic_embeds", acoustic_embeds.shape, 'FP32'),
        InferInput("acoustic_embeds_len", acoustic_embeds_len.shape, 'INT32')
    ]
    inputs[0].set_data_from_numpy(enc)
    inputs[1].set_data_from_numpy(enc_len)
    inputs[2].set_data_from_numpy(acoustic_embeds)
    inputs[3].set_data_from_numpy(acoustic_embeds_len)
    outputs = [InferRequestedOutput('logits'), InferRequestedOutput('sample_ids')]
    response = client.infer(
        model_name="decoder",
        inputs=inputs,
        outputs=outputs,
        sequence_id=1234321,
        sequence_start=i == 0,
        sequence_end=i == total_chunk_num - 1,
    )
    # transcripts = response.as_numpy('transcripts')
    logits = response.as_numpy('logits')
    sample_ids = response.as_numpy('sample_ids')
    print('logits: ', logits, sample_ids)
