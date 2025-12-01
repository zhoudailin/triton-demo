import numpy as np
from tritonclient.grpc import InferenceServerClient, InferInput, InferRequestedOutput

client = InferenceServerClient(url="localhost:8001")

enc = np.random.randn(1, 10, 512).astype(np.float32)
enc_len = np.array([10], dtype=np.int32)
acoustic = np.random.randn(1, 3, 512).astype(np.float32)
acoustic_len = np.array([3], dtype=np.int32)

inputs = [
    InferInput("enc", enc.shape, "FP32"),
    InferInput("enc_len", enc_len.reshape(1, 1).shape, "INT32"),
    InferInput("acoustic_embeds", acoustic.shape, "FP32"),
    InferInput("acoustic_embeds_len", acoustic_len.reshape(1, 1).shape, "INT32"),
]
inputs[0].set_data_from_numpy(enc)
inputs[1].set_data_from_numpy(enc_len.reshape(1, 1))
inputs[2].set_data_from_numpy(acoustic)
inputs[3].set_data_from_numpy(acoustic_len.reshape(1, 1))

outputs = [InferRequestedOutput("sample_ids")]
response = client.infer(
    model_name="decoder",
    inputs=inputs,
    outputs=outputs,
    sequence_id=12345,
    sequence_start=True,
    sequence_end=True,
)
print("sample_ids:", response.as_numpy("sample_ids"))
