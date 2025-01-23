import torch
import outlines
from vllm.sampling_params import SamplingParams

# Clients
class Client:
    def __init__(self, model_name, model_type):
        self.model_name = model_name
        self.model_type = model_type

        if self.model_type == "openai":
            self.model = OpenAIClient(self.model_name)
        elif self.model_type == "hf":
            self.model = HFClient(self.model_name)
        else:
            raise ValueError(f"Invalid model type: {self.model_type}")

    def text_generate(self, prompt, k=1, temperature=0.8):
        params = SamplingParams(max_tokens=8192)
        text_generator = outlines.generate.text(self.model, sampler=outlines.samplers.multinomial(k, temperature=temperature))
        return text_generator(prompt, sampling_params=params)
    
class OpenAIClient(Client):
    def __init__(self, model_name):
        self.model_name = model_name
        self.api_key = "[OPENAI_KEY]"
        self.model = outlines.models.openai(self.model_name, api_key=self.api_key)

    def json_generate(self, json_schema, prompt, temperature=0.8):
        json_generator = outlines.generate.json(self.model, json_schema, sampler=outlines.samplers.multinomial(3, temperature=temperature))
        return json_generator(prompt)

class HFClient(Client):
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = outlines.models.vllm(self.model_name, max_model_len=16384, gpu_memory_utilization=0.8, tensor_parallel_size=torch.cuda.device_count())
    
    def json_generate(self, json_schema, prompt, k=1, temperature=0.8):
        params = SamplingParams(max_tokens=16384)
        json_generator = outlines.generate.json(self.model, json_schema, sampler=outlines.samplers.multinomial(k, temperature=temperature))
        return json_generator(prompt, sampling_params=params)