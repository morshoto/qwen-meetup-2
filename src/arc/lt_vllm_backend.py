"""vLLM implementation of the registered direct-grid protocol."""
import json

from src.arc.lt_direct_format import normalize_direct
from src.arc.lt_live_backend import PROMPT

VLLM_SETTINGS = {'dtype': 'bfloat16', 'max_model_len': 16384, 'tensor_parallel_size': 1,
                 'language_model_only': True, 'enforce_eager': True, 'max_num_seqs': 1}


class VLLMBackend:
    def __init__(self, model_path, *, seed=20261111, max_tokens=4096, temperature=0.6, max_model_len=None):
        from vllm import LLM, SamplingParams
        settings = dict(VLLM_SETTINGS)
        if max_model_len is not None:
            if type(max_model_len) is not int or not 1 <= max_model_len <= 131072:
                raise ValueError('invalid context length')
            settings['max_model_len'] = max_model_len
            settings['gpu_memory_utilization'] = 0.95
        self.llm = LLM(model=str(model_path), trust_remote_code=False, **settings)
        self.tokenizer = self.llm.get_tokenizer()
        self.max_model_len = max_model_len
        self.sampling_params = SamplingParams
        self.seed, self.max_tokens, self.temperature = seed, max_tokens, temperature
        self.raw = self.stats = None

    def generate(self, prompt, *, kind, candidate):
        if kind != 'direct' or candidate not in (0, 1):
            raise ValueError('invalid request')
        payload = json.loads(prompt)
        count = len(payload['task']['test'])
        payload['instruction'] = PROMPT
        prompt = json.dumps(payload, sort_keys=True, separators=(',', ':'))
        formatted = self.tokenizer.apply_chat_template(
            [{'role': 'user', 'content': prompt}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        if self.max_model_len is not None:
            prompt_tokens = len(self.tokenizer.encode(formatted, add_special_tokens=False))
            if prompt_tokens + self.max_tokens > self.max_model_len:
                raise ValueError('prompt and output budget exceed registered context')
        params = self.sampling_params(temperature=self.temperature, top_p=1.0, top_k=-1,
                                      seed=self.seed + candidate, max_tokens=self.max_tokens)
        response = self.llm.generate([formatted], params, use_tqdm=False)[0]
        output = response.outputs[0]
        self.raw = output.text
        self.stats = {'prompt_tokens': len(response.prompt_token_ids),
                      'completion_tokens': len(output.token_ids), 'peak_memory_gb': None,
                      'finish_reason': output.finish_reason}
        try:
            return json.dumps(normalize_direct(self.raw, count), separators=(',', ':'))
        except ValueError:
            return 'null'
