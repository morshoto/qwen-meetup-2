"""vLLM implementation of the registered direct-grid protocol."""
import json

from src.arc.lt_direct_format import normalize_direct
from src.arc.lt_live_backend import PROMPT


class VLLMBackend:
    def __init__(self, model_path, *, seed=20261111, max_tokens=4096, temperature=0.6):
        from vllm import LLM, SamplingParams
        self.llm = LLM(model=str(model_path), dtype='bfloat16', max_model_len=16384,
                       trust_remote_code=False, tensor_parallel_size=1)
        self.tokenizer = self.llm.get_tokenizer()
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
