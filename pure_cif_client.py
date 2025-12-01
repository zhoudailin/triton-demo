import numpy as np
from tritonclient.grpc import InferenceServerClient, InferInput, InferRequestedOutput

client = InferenceServerClient(url="localhost:8001")

enc = np.random.randn(10, 512).astype(np.float32)
enc_len = np.array([10], dtype=np.int32)
acoustic = np.random.randn(1, 3, 512).astype(np.float32)
acoustic_len = np.array([3], dtype=np.int32)

inputs = [
    InferInput("enc", enc.shape, "FP32"),
    InferInput("enc_len", enc_len.shape, "INT32"),
    InferInput("acoustic_embeds", acoustic.reshape(-1, 512).shape, "FP32"),
    InferInput("acoustic_embeds_len", acoustic_len.shape, "INT32"),
]
inputs[0].set_data_from_numpy(enc)
inputs[1].set_data_from_numpy(enc_len)
inputs[2].set_data_from_numpy(acoustic.reshape(-1, 512))
inputs[3].set_data_from_numpy(acoustic_len)

outputs = [InferRequestedOutput("transcripts")]

response = client.infer(
    model_name="pure_cif",
    inputs=inputs,
    outputs=outputs,
)
print("transcripts:", response.as_numpy("transcripts"))
