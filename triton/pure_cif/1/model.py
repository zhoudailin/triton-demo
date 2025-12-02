import json
import traceback

import numpy as np
import triton_python_backend_utils as pb_utils
try:
    import torch
except ImportError:
    torch = None


class TritonPythonModel:
    def initialize(self, args):
        model_config = json.loads(args["model_config"])
        vocab_path = model_config["parameters"]["vocabulary"]["string_value"]
        try:
            with open(vocab_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            if isinstance(config, dict) and "token_list" in config:
                self.tokens = config["token_list"]
            else:
                self.tokens = config
        except Exception:
            traceback.print_exc()
            self.tokens = []

    def _tensor_to_numpy(self, tensor):
        if tensor is None:
            return None
        try:
            return tensor.as_numpy()
        except Exception:
            # Some backends return GPU tensors; fall back to DLPack -> torch -> CPU.
            if torch is None:
                traceback.print_exc()
                return None
            try:
                return torch.utils.dlpack.from_dlpack(tensor.to_dlpack()).cpu().numpy()
            except Exception:
                traceback.print_exc()
                return None

    def execute(self, requests):
        responses = []
        for request in requests:
            enc = pb_utils.get_input_tensor_by_name(request, "enc")
            enc_np = enc.as_numpy()
            enc_len = pb_utils.get_input_tensor_by_name(request, "enc_len").as_numpy()
            acoustic = pb_utils.get_input_tensor_by_name(request, "acoustic_embeds").as_numpy()
            acoustic_len = pb_utils.get_input_tensor_by_name(request, "acoustic_embeds_len").as_numpy()

            if enc_np.ndim == 2:
                enc_np = enc_np.reshape(1, enc_np.shape[0], enc_np.shape[1])
            if acoustic.ndim == 2:
                acoustic = acoustic.reshape(1, acoustic.shape[0], acoustic.shape[1])

            input_tensors = [
                pb_utils.Tensor("enc", enc_np.astype(np.float32)),
                pb_utils.Tensor("enc_len", enc_len.astype(np.int32).reshape(1, 1)),
                pb_utils.Tensor("acoustic_embeds", acoustic.astype(np.float32)),
                pb_utils.Tensor(
                    "acoustic_embeds_len", acoustic_len.astype(np.int32).reshape(1, 1)
                ),
            ]

            inference_request = pb_utils.InferenceRequest(
                model_name="decoder",
                requested_output_names=["sample_ids", "logits"],
                inputs=input_tensors,
                correlation_id=12345,
                flags=3,
            )
            inference_response = inference_request.exec()
            if inference_response.has_error():
                err = inference_response.error().message()
                print(f"Decoder call failed: {err}")
                transcript = ""
            else:
                sample_ids = pb_utils.get_output_tensor_by_name(
                    inference_response, "sample_ids"
                )
                logits = pb_utils.get_output_tensor_by_name(
                    inference_response, "logits"
                )
                sample_ids_np = self._tensor_to_numpy(sample_ids)
                logits_np = self._tensor_to_numpy(logits)
                print(
                    "decoder outputs: sample_ids shape",
                    None if sample_ids_np is None else sample_ids_np.shape,
                    "logits shape",
                    None if logits_np is None else logits_np.shape,
                )
                if sample_ids_np is None:
                    print("decoder returned None sample_ids")
                    transcript = ""
                else:
                    token_ids = sample_ids_np[0]
                    print("sample_ids array:", token_ids)
                    tokens = []
                    for tid in token_ids:
                        idx = int(tid)
                        if 0 <= idx < len(self.tokens):
                            tokens.append(self.tokens[idx])
                        else:
                            print(f"token id {idx} out of range")
                    transcript = "".join(tokens)
                    print("transcript candidate:", transcript)

            out = pb_utils.Tensor(
                "transcripts", np.array([transcript], dtype=object)
            )
            responses.append(pb_utils.InferenceResponse(output_tensors=[out]))

        return responses
