from funasr import AutoModel

model = AutoModel(
    model="paraformer-zh", model_revision="v2.0.4",
    vad_model="fsmn-vad", vad_model_revision="v2.0.4"
)
res = model.generate(input=f"{model.model_path}/example/asr_example.wav",
                     batch_size_s=300,
                     hotword='魔搭')
print(res)
