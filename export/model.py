from funasr import AutoModel
from funasr_onnx import Paraformer

# model = AutoModel(
#     model="paraformer-zh-streaming",
#     disable_update=True
# )
# model.export(output_dir="model")
model = Paraformer(
    model_dir="iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online"
)
res = model("/home/mrzhou/Documents/codes/private/triton-demo/data/asr_example.wav")
print(res)
