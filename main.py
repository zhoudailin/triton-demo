import math
import time

import tritonclient.grpc as grpcclient
import wave
from pathlib import Path

import numpy as np

client = grpcclient.InferenceServerClient("localhost:8001")


def load_wav(path: Path) -> np.ndarray:
    """Load PCM16 mono wav and return shape (1, num_samples) float32 array."""
    with wave.open(str(path), "rb") as wf:
        if wf.getsampwidth() != 2:
            raise ValueError(f"{path} must be 16-bit PCM.")
        frames = wf.readframes(wf.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
        audio /= np.iinfo(np.int16).max
        if wf.getnchannels() > 1:
            audio = audio.reshape(-1, wf.getnchannels()).mean(axis=1)
    return np.expand_dims(audio, axis=0)


wav_path = Path("data/asr_example.wav")
wav = load_wav(wav_path)
total_samples = wav.shape[1]
print("Loaded wav:", wav.shape, total_samples)

SAMPLE_RATE = 16000
CHUNK_SIZE_S = 0.6
chunk_size = int(SAMPLE_RATE * CHUNK_SIZE_S)
num_chunks = max(1, math.ceil(total_samples / chunk_size))
sequence_id = int(time.time() * 1000)

final_transcript = None
for chunk_idx in range(num_chunks):
    start = chunk_idx * chunk_size
    end = min(total_samples, start + chunk_size)
    chunk = wav[:, start:end]
    inputs = [grpcclient.InferInput("wav", chunk.shape, "FP32")]
    inputs[0].set_data_from_numpy(chunk)
    outputs = [grpcclient.InferRequestedOutput("transcripts")]
    response = client.infer(
        model_name="streaming_paraformer",
        inputs=inputs,
        outputs=outputs,
        sequence_id=sequence_id,
        sequence_start=chunk_idx == 0,
        sequence_end=chunk_idx == num_chunks - 1,
    )
    transcripts = response.as_numpy("transcripts")
    final_transcript = transcripts
    print(f"chunk {chunk_idx + 1}/{num_chunks} -> {transcripts}")

if final_transcript is not None:
    print("Final transcripts:", final_transcript, final_transcript.shape)
