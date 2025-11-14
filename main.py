import tritonclient.grpc as grpcclient
import numpy as np

client = grpcclient.InferenceServerClient("localhost:8001")

wav = np.random.rand(1, 16000).astype(np.float32)
wav_lens = np.array([len(wav)], dtype=np.int32).reshape(1, 1)

print(wav, wav.shape, wav_lens, wav_lens.shape)

inputs = [
    grpcclient.InferInput("wav", wav.shape, "FP32"),
]
inputs[0].set_data_from_numpy(wav)

outputs = [grpcclient.InferRequestedOutput("speech")]

response = client.infer(
    model_name="feature_extractor",
    inputs=inputs,
    outputs=outputs,
    sequence_id=1234,
    sequence_start=True,
    sequence_end=False,
)
speech = response.as_numpy("speech")
print("Speech shape:", speech, speech.shape)
