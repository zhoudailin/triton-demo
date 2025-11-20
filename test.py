from funasr import AutoModel
import tritonclient.grpc as grpcclient
import numpy as np
import time
import os
import soundfile

# 配置参数
chunk_size = [0, 10, 5]  # [0, 10, 5] 600ms, [0, 8, 4] 480ms
encoder_chunk_look_back = 4  # number of chunks to lookback for encoder self-attention
decoder_chunk_look_back = 1  # number of encoder chunks to lookback for decoder cross-attention

try:
    model = AutoModel(model="paraformer-zh-streaming", model_revision="v2.0.4", disable_update=True)
    print("FunASR model loaded successfully")
except Exception as e:
    print(f"Failed to load FunASR model: {e}")
    exit(1)

try:
    client = grpcclient.InferenceServerClient("localhost:8001")
    # 检查服务器连接
    if not client.is_server_live():
        print("Triton server is not live")
        exit(1)
    if not client.is_server_ready():
        print("Triton server is not ready")
        exit(1)
    print("Triton server connection successful")
except Exception as e:
    print(f"Failed to connect to Triton server: {e}")
    exit(1)

try:
    wav_file = os.path.join(model.model_path, "example/asr_example.wav")
    if not os.path.exists(wav_file):
        print(f"Audio file not found: {wav_file}")
        # 使用默认的测试音频文件
        wav_file = "data/asr_example.wav"
        if not os.path.exists(wav_file):
            print("Default audio file not found, creating a test signal...")
            # 创建一个简单的测试信号
            duration = 3.0  # 3秒
            sample_rate = 16000
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            # 生成一个440Hz的正弦波
            speech = 0.3 * np.sin(2 * np.pi * 440 * t)
            # 保存为临时文件
            os.makedirs("data", exist_ok=True)
            soundfile.write(wav_file, speech, sample_rate)
        else:
            speech, sample_rate = soundfile.read(wav_file)
    else:
        speech, sample_rate = soundfile.read(wav_file)

    print(f"Loaded audio: {wav_file}, sample_rate: {sample_rate}, duration: {len(speech)/sample_rate:.2f}s")

    # 确保采样率为16kHz
    if sample_rate != 16000:
        print(f"Warning: Audio sample rate is {sample_rate}, expected 16000")
        # 可以在这里添加重采样逻辑

    # 转换为float32格式
    if speech.dtype != np.float32:
        speech = speech.astype(np.float32)
        print("Converted audio to float32 format")

except Exception as e:
    print(f"Failed to load audio: {e}")
    exit(1)

chunk_stride = chunk_size[1] * 960  # 600ms
total_chunk_num = int(len(speech) / chunk_stride) + 1
sequence_id = int(time.time() * 1000)  # 使用时间戳作为序列ID避免冲突

print(f"Total chunks: {total_chunk_num}, chunk_stride: {chunk_stride}")
print(f"Starting inference with sequence_id: {sequence_id}")

cache = {}
all_results = []

try:
    for i in range(total_chunk_num):
        start_idx = i * chunk_stride
        end_idx = min((i + 1) * chunk_stride, len(speech))
        speech_chunk = speech[start_idx:end_idx]
        is_final = i == total_chunk_num - 1

        print(f"\nProcessing chunk {i+1}/{total_chunk_num}, shape: {speech_chunk.shape}, is_final: {is_final}")

        # 确保输入形状正确 (batch_size=1, sequence_length)
        if speech_chunk.ndim == 1:
            speech_chunk = np.expand_dims(speech_chunk, axis=0)

        try:
            inputs = [grpcclient.InferInput("wav", speech_chunk.shape, "FP32")]
            inputs[0].set_data_from_numpy(speech_chunk.astype(np.float32))
            outputs = [grpcclient.InferRequestedOutput("transcripts")]

            response = client.infer(
                model_name="streaming_paraformer",
                inputs=inputs,
                outputs=outputs,
                sequence_id=sequence_id,
                sequence_start=i == 0,
                sequence_end=is_final,
            )

            transcripts = response.as_numpy("transcripts")
            print(f"Chunk {i+1} result: {transcripts}")

            if transcripts.size > 0:
                result_text = transcripts[0].decode('utf-8') if isinstance(transcripts[0], bytes) else str(transcripts[0])
                all_results.append(result_text)
                print(f"Decoded text: {result_text}")

        except Exception as e:
            print(f"Error processing chunk {i+1}: {e}")
            continue

except Exception as e:
    print(f"Error during inference: {e}")

print(f"\nAll results: {all_results}")
if all_results:
    final_result = "".join(all_results)
    print(f"Final transcription: {final_result}")

print("Inference completed")
