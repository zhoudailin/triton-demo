from tritonclient.grpc import InferenceServerClient, InferInput, InferRequestedOutput
import numpy as np

client = InferenceServerClient(url="localhost:8001")

speech = np.random.randn(560).reshape(1, 1, 560).astype(np.float32)
speech_lengths = np.array([[560]], dtype=np.int32)

inputs = [
    InferInput('speech', speech.shape, 'FP32'),
    InferInput('speech_lengths', speech_lengths.shape, 'INT32'),
]

inputs[0].set_data_from_numpy(speech)
inputs[1].set_data_from_numpy(speech_lengths)

outputs = [InferRequestedOutput('enc')]
response = client.infer(
    model_name="bls-demo",
    inputs=inputs,
    outputs=outputs
)
enc = response.as_numpy('enc')
