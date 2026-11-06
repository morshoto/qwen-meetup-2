"""Native MLX implementation of the frozen LT direct-grid prompt."""
import json
from src.arc.lt_direct_format import normalize_direct

PROMPT = ('Return only JSON: an array of grids, one per test input. '
          'Syntax-only example for one output: [[[0,1],[2,3]]]. '
          'Do not copy the example. Compute the output for every actual test input. '
          'Each grid is an array of integer rows. No object wrappers, markdown, or explanation.')


class MLXBackend:
    def __init__(self, model_path, *, seed=20261111, max_tokens=4096, temperature=0.6):
        import mlx.core as mx
        from mlx_lm import load, stream_generate
        from mlx_lm.sample_utils import make_sampler
        self.mx, self.stream_generate, self.make_sampler = mx, stream_generate, make_sampler
        self.model, self.tokenizer = load(str(model_path))
        self.seed, self.max_tokens, self.temperature = seed, max_tokens, temperature
        self.raw = self.stats = None

    def generate(self, prompt, *, kind, candidate):
        if kind != 'direct' or candidate not in (0, 1): raise ValueError('invalid request')
        payload = json.loads(prompt)
        count = len(payload['task']['test'])
        payload['instruction'] = PROMPT
        prompt = json.dumps(payload, sort_keys=True, separators=(',', ':'))
        formatted = self.tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False,
                                                       add_generation_prompt=True, enable_thinking=False)
        self.mx.random.seed(self.seed + candidate)
        self.raw = ''; self.stats = None
        for response in self.stream_generate(self.model, self.tokenizer, prompt=formatted,
                max_tokens=self.max_tokens, sampler=self.make_sampler(temp=self.temperature)):
            self.raw += response.text
            self.stats = {'prompt_tokens': response.prompt_tokens, 'completion_tokens': response.generation_tokens,
                          'peak_memory_gb': response.peak_memory, 'finish_reason': response.finish_reason}
        try: return json.dumps(normalize_direct(self.raw, count), separators=(',', ':'))
        except ValueError: return 'null'
