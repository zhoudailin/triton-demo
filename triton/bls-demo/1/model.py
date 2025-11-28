import triton_python_backend_utils as pb_utils
from torch import from_dlpack


class TritonPythonModel:
    def execute(self, requests):
        responses = []
        for request in requests:
            input_speech = pb_utils.get_input_tensor_by_name(request, "speech")
            input_speech_lengths = pb_utils.get_input_tensor_by_name(request, "speech_lengths")

            input_tensor0 = pb_utils.Tensor("speech", input_speech.as_numpy())
            input_tensor1 = pb_utils.Tensor("speech_lengths", input_speech_lengths.as_numpy())
            inputs = [input_tensor0, input_tensor1]

            inference_request = pb_utils.InferenceRequest(
                model_name='encoder',
                requested_output_names=['enc', 'enc_len', 'alphas'],
                inputs=inputs
            )
            inference_response = inference_request.exec()
            print(inference_response)
            return [inference_response]
