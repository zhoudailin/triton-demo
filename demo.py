import math
import os
import sys

repo_dir = os.path.dirname(os.path.abspath(__file__))
funasr_src = os.path.join(repo_dir, "funasr-source")
if funasr_src not in sys.path:
    sys.path.insert(0, funasr_src)

from funasr import AutoModel

chunk_size = [0, 10, 5]  # [0, 10, 5] 600ms, [0, 8, 4] 480ms
encoder_chunk_look_back = 5  # number of chunks to lookback for encoder self-attention
decoder_chunk_look_back = 2  # number of encoder chunks to lookback for decoder cross-attention

model = AutoModel(model="paraformer-zh-streaming", model_revision="v2.0.4")

import soundfile

wav_file = os.path.join("data/asr_example.wav")
speech, sample_rate = soundfile.read(wav_file)
chunk_stride = chunk_size[1] * 960  # 600ms

cache = {}
total_chunk_num = math.ceil(len(speech) / chunk_stride)
print(f'total_chunk_num: {total_chunk_num}')
for i in range(total_chunk_num):
    speech_chunk = speech[i * chunk_stride:(i + 1) * chunk_stride]
    is_final = i == total_chunk_num - 1
    response = model.generate(
        input=speech_chunk,
        is_final=is_final,
        cache=cache,
        encoder_chunk_look_back = encoder_chunk_look_back,
        decoder_chunk_look_back = decoder_chunk_look_back
    )
    print(response)
