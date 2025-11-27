from funasr import AutoModel

online_model = AutoModel(model="paraformer-zh-streaming")
online_model.export(output_dir="exports_model/online")

seaco_model = AutoModel(
    model="iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
)
seaco_model.export(output_dir="exports_model/seaco")

vad_model = AutoModel(model="fsmn-vad")
vad_model.export(output_dir="exports_model/vad")

punc_model = AutoModel(model="ct-punc")
punc_model.export(output_dir="exports_model/punc")
