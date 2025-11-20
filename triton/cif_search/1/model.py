import asyncio
import json
from collections import OrderedDict

import numpy as np
import triton_python_backend_utils as pb_utils
import yaml
from torch.utils.dlpack import from_dlpack


class LimitedDict(OrderedDict):
    def __init__(self, max_length):
        super().__init__()
        self.max_length = max_length

    def __setitem__(self, key, value):
        if len(self) >= self.max_length:
            self.popitem(last=False)
        super().__setitem__(key, value)


class CIFSearch:
    """CIFSearch: https://github.com/alibaba-damo-academy/FunASR/blob/main/runtime/python/onnxruntime/funasr_onnx
    /paraformer_online_bin.py"""

    def __init__(self):
        self.cache = {
            "cif_hidden": np.zeros((1, 1, 512)).astype(np.float32),
            "cif_alphas": np.zeros((1, 1)).astype(np.float32),
            "last_chunk": False,
        }
        self.chunk_size = [5, 10, 5]
        self.tail_threshold = 0.45
        self.cif_threshold = 1.0

    def infer(self, hidden, alphas):
        batch_size, len_time, hidden_size = hidden.shape
        token_length = []
        list_fires = []
        list_frames = []
        cache_alphas = []
        cache_hiddens = []
        alphas[:, : self.chunk_size[0]] = 0.0
        alphas[:, sum(self.chunk_size[:2]):] = 0.0

        if self.cache is not None and "cif_alphas" in self.cache and "cif_hidden" in self.cache:
            hidden = np.concatenate((self.cache["cif_hidden"], hidden), axis=1)
            alphas = np.concatenate((self.cache["cif_alphas"], alphas), axis=1)
        if self.cache is not None and "last_chunk" in self.cache and self.cache["last_chunk"]:
            tail_hidden = np.zeros((batch_size, 1, hidden_size)).astype(np.float32)
            tail_alphas = np.array([[self.tail_threshold]]).astype(np.float32)
            tail_alphas = np.tile(tail_alphas, (batch_size, 1))
            hidden = np.concatenate((hidden, tail_hidden), axis=1)
            alphas = np.concatenate((alphas, tail_alphas), axis=1)

        len_time = alphas.shape[1]
        for b in range(batch_size):
            integrate = 0.0
            frames = np.zeros(hidden_size).astype(np.float32)
            list_frame = []
            list_fire = []
            for t in range(len_time):
                alpha = alphas[b][t]
                if alpha + integrate < self.cif_threshold:
                    integrate += alpha
                    list_fire.append(integrate)
                    frames += alpha * hidden[b][t]
                else:
                    frames += (self.cif_threshold - integrate) * hidden[b][t]
                    list_frame.append(frames)
                    integrate += alpha
                    list_fire.append(integrate)
                    integrate -= self.cif_threshold
                    frames = integrate * hidden[b][t]

            cache_alphas.append(integrate)
            if integrate > 0.0:
                cache_hiddens.append(frames / integrate)
            else:
                cache_hiddens.append(frames)

            token_length.append(len(list_frame))
            list_fires.append(list_fire)
            list_frames.append(list_frame)

        max_token_len = max(token_length)
        list_ls = []
        for b in range(batch_size):
            pad_frames = np.zeros((max_token_len - token_length[b], hidden_size)).astype(np.float32)
            if token_length[b] == 0:
                list_ls.append(pad_frames)
            else:
                list_ls.append(np.concatenate((list_frames[b], pad_frames), axis=0))

        self.cache["cif_alphas"] = np.stack(cache_alphas, axis=0)
        self.cache["cif_alphas"] = np.expand_dims(self.cache["cif_alphas"], axis=0)
        self.cache["cif_hidden"] = np.stack(cache_hiddens, axis=0)
        self.cache["cif_hidden"] = np.expand_dims(self.cache["cif_hidden"], axis=0)

        return np.stack(list_ls, axis=0).astype(np.float32), np.stack(token_length, axis=0).astype(
            np.int32
        )


class TritonPythonModel:
    def initialize(self, args):
        self.model_config = model_config = json.loads(args["model_config"])
        self.max_batch_size = max(model_config["max_batch_size"], 1)

        # # Get OUTPUT0 configuration
        output0_config = pb_utils.get_output_config_by_name(model_config, "transcripts")
        # # Convert Triton types to numpy types
        self.out0_dtype = pb_utils.triton_string_to_numpy(output0_config["data_type"])

        self.init_vocab(self.model_config["parameters"])

        self.cif_search_cache = LimitedDict(1024)
        self.start = LimitedDict(1024)

    def init_vocab(self, parameters):
        for li in parameters.items():
            key, value = li
            value = value["string_value"]
            if key == "vocabulary":
                self.vocab_dict = self.load_vocab(value)

    def load_vocab(self, vocab_file):
        print(f"Loading vocab from: {vocab_file}")
        try:
            with open(str(vocab_file), "rb") as f:
                config = yaml.load(f, Loader=yaml.Loader)
            vocab_list = config["token_list"]
            print(f"Loaded vocab with {len(vocab_list)} tokens")
            return vocab_list
        except Exception as e:
            print(f"Error loading vocab: {e}")
            return {}

    async def execute(self, requests):
        print(f"CIF Search execute: received {len(requests)} requests")
        batch_end = []
        responses = []
        batch_corrid = []
        qualified_corrid = []
        batch_result = {}
        inference_response_awaits = []

        for request in requests:
            hidden = pb_utils.get_input_tensor_by_name(request, "enc")
            hidden = hidden.as_numpy()
            alphas = pb_utils.get_input_tensor_by_name(request, "alphas")
            alphas = alphas.as_numpy()
            hidden_len = pb_utils.get_input_tensor_by_name(request, "enc_len")
            hidden_len = hidden_len.as_numpy()

            in_start = pb_utils.get_input_tensor_by_name(request, "START")
            start = in_start.as_numpy()[0][0]

            in_corrid = pb_utils.get_input_tensor_by_name(request, "CORRID")
            corrid = in_corrid.as_numpy()[0][0]

            in_end = pb_utils.get_input_tensor_by_name(request, "END")
            end = in_end.as_numpy()[0][0]

            batch_end.append(end)
            batch_corrid.append(corrid)

            if start:
                self.cif_search_cache[corrid] = CIFSearch()
                self.start[corrid] = 1
            if end and corrid in self.cif_search_cache:
                self.cif_search_cache[corrid].cache["last_chunk"] = True

            try:
                acoustic, acoustic_len = self.cif_search_cache[corrid].infer(hidden, alphas)
                batch_result[corrid] = ""
                if acoustic.shape[1] == 0:
                    continue
            except Exception as e:
                print(f"CIF inference error for corrid {corrid}: {e}")
                batch_result[corrid] = ""
                continue

            qualified_corrid.append(corrid)
            # 修复数据类型和形状问题
            input_tensor0 = pb_utils.Tensor("enc", hidden.astype(np.float32))
            # hidden_len已经是标量，直接使用
            input_tensor1 = pb_utils.Tensor("enc_len", hidden_len.astype(np.int32))
            input_tensor2 = pb_utils.Tensor("acoustic_embeds", acoustic.astype(np.float32))
            input_tensor3 = pb_utils.Tensor(
                "acoustic_embeds_len", acoustic_len.astype(np.int32)
            )
            input_tensors = [input_tensor0, input_tensor1, input_tensor2, input_tensor3]

            if self.start[corrid] and end:
                flag = 3
            elif end:
                flag = 2
            elif self.start[corrid]:
                flag = 1
                self.start[corrid] = 0
            else:
                flag = 0

            inference_request = pb_utils.InferenceRequest(
                model_name="decoder",
                requested_output_names=["sample_ids"],
                inputs=input_tensors,
                request_id="",
                correlation_id=corrid,
                flags=flag,
            )
            inference_response_awaits.append(inference_request.async_exec())

        inference_responses = []
        if inference_response_awaits:
            # 修复异步调用问题
            for i, future in enumerate(inference_response_awaits):
                try:
                    response = await future
                    if response.has_error():
                        print(f"Inference failed for qualified_corrid[{i}] {qualified_corrid[i]}: {response.error().message()}")
                        inference_responses.append(None)
                    else:
                        inference_responses.append(response)
                except Exception as e:
                    print(f"Async inference error for qualified_corrid[{i}] {qualified_corrid[i]}: {e}")
                    inference_responses.append(None)

        for index_corrid, inference_response in zip(qualified_corrid, inference_responses):
            if inference_response is None:
                print(f"Skipping corrid {index_corrid} due to inference failure")
                continue

            try:
                sample_ids = pb_utils.get_output_tensor_by_name(inference_response, "sample_ids")
                token_ids = sample_ids.as_numpy()[0]
                # 确保token_ids在词汇表范围内
                tokens = []
                for token_id in token_ids:
                    if int(token_id) in self.vocab_dict:
                        tokens.append(self.vocab_dict[int(token_id)])
                    else:
                        print(f"Warning: token_id {token_id} not in vocab")
                batch_result[index_corrid] = "".join(tokens)
                print(f"Generated text for corrid {index_corrid}: {batch_result[index_corrid]}")
            except Exception as e:
                print(f"Error processing inference response for corrid {index_corrid}: {e}")
                continue

        for i, index_corrid in enumerate(batch_corrid):
            sent = np.array([batch_result[index_corrid]])
            out0 = pb_utils.Tensor("transcripts", sent.astype(self.out0_dtype))
            inference_response = pb_utils.InferenceResponse(output_tensors=[out0])
            responses.append(inference_response)

            if batch_end[i]:
                del self.cif_search_cache[index_corrid]
                del self.start[index_corrid]

        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print("Cleaning up...")
