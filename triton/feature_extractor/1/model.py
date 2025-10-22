import json
import triton_python_backend_utils as pb_utils
class TritonPythonModel:
    def __init__(self):
        self.output_dtype = None
        self.device = None
        self.model_config = None

    def initialize(self, args):
        self.model_config = model_config = json.loads(args["model_config"])

        if args['model_instance_kind'] == 'GPU':
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        output_config = pb_utils.get_output_config_by_name(model_config, 'speech')
        self.output_dtype = pb_utils.triton_string_to_numpy(output_config['data_type'])

        parameters = self.model_config['parameters']
        config_path = parameters['config_path']['string_value']
        chunk_size_s = parameters['chunk_size_s']['string_value']
        print(f'config_path: {config_path} chunk_size_s: {chunk_size_s}')


    def execute(self, requests):
        pass

    def finalize(self):
        pass
