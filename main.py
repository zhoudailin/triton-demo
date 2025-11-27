import math

import numpy as np
from tritonclient.grpc import InferenceServerClient, InferInput, InferRequestedOutput

from funasr import AutoModel

chunk_size = [0, 10, 5]  # [0, 10, 5] 600ms, [0, 8, 4] 480ms
encoder_chunk_look_back = 5  # number of chunks to lookback for encoder self-attention
decoder_chunk_look_back = 2  # number of encoder chunks to lookback for decoder cross-attention

model = AutoModel(model="paraformer-zh-streaming", model_revision="v2.0.4")

import soundfile
import os

client = InferenceServerClient(url="localhost:8001")

wav_file = os.path.join(model.model_path, "example/asr_example.wav")
speech, sample_rate = soundfile.read(wav_file)
chunk_stride = chunk_size[1] * 960  # 600ms

cache = {}
total_chunk_num = math.ceil(len(speech) / chunk_stride)
print(f'total_chunk_num: {total_chunk_num}')
for i in range(total_chunk_num):
    speech_chunk = speech[i * chunk_stride:(i + 1) * chunk_stride]
    is_final = i == total_chunk_num - 1
    # res = model.generate(input=speech_chunk, cache=cache, is_final=is_final, chunk_size=chunk_size,
    #                      encoder_chunk_look_back=encoder_chunk_look_back,
    #                      decoder_chunk_look_back=decoder_chunk_look_back)
    wav_np = speech_chunk.astype(np.float32).reshape(1, -1)
    inputs = [
        InferInput('wav', wav_np.shape, 'FP32'),
    ]
    inputs[0].set_data_from_numpy(wav_np)
    outputs = [InferRequestedOutput('chunk_xs_out')]
    response = client.infer(
        model_name="streaming_paraformer",
        inputs=inputs,
        outputs=outputs,
        sequence_id=1234321,
        sequence_start=i == 0,
        sequence_end=i == total_chunk_num - 1,
    )
    speech = response.as_numpy('chunk_xs_out')
    print(speech, speech.shape)
