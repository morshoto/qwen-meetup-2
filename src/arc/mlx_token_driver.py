"""Optional local MLX-LM token driver for the guarded generation boundary.

Loading is explicit and only accepts an existing local directory. No MLX-LM or
model dependency is imported in CPU fake mode. A trusted adapter must run this
in the same process as GuardedGenerator; it must not expose the driver directly.
"""
from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
from typing import Any, Iterator


def adapter_sha256(adapter_path: str | Path) -> str:
    """Hash the two MLX-LM files that define an adapter and its structure."""
    path = Path(adapter_path).expanduser().resolve()
    config = path / 'adapter_config.json'
    weights = path / 'adapters.safetensors'
    if not path.is_dir() or not config.is_file() or not weights.is_file():
        raise ValueError('adapter must contain adapter_config.json and adapters.safetensors')
    digest = hashlib.sha256()
    for item in (config, weights):
        digest.update(item.name.encode('utf-8') + b'\0')
        with item.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
    return digest.hexdigest()


class MLXTokenDriver:
    def __init__(self, model_dir: str | Path, *, temperature: float, seed: int,
                 thinking: str = 'server_default', adapter_path: str | Path | None = None,
                 expected_adapter_sha256: str = 'none') -> None:
        path = Path(model_dir).expanduser().resolve()
        if not path.is_dir():
            raise ValueError('MLX model must be an existing local directory')
        if not 0 <= temperature <= 2:
            raise ValueError('invalid temperature')
        try:
            import mlx.core as mx
            from mlx_lm.generate import generate_step
            from mlx_lm.sample_utils import make_sampler
            from mlx_lm.utils import load
        except ImportError as exc:
            raise RuntimeError('MLX-LM requires an installed native Apple Silicon environment') from exc
        try:
            self.generate_step_sha256 = hashlib.sha256(inspect.getsource(generate_step).encode('utf-8')).hexdigest()
        except (OSError, TypeError) as exc:
            raise RuntimeError('cannot inspect MLX-LM generation source') from exc
        self._mx, self._generate_step = mx, generate_step
        if adapter_path is None:
            if expected_adapter_sha256 != 'none':
                raise ValueError('expected adapter hash was supplied without an adapter path')
            self.adapter_sha256 = 'none'
            self._model, self._tokenizer = load(str(path))
        else:
            adapter = Path(adapter_path).expanduser().resolve()
            self.adapter_sha256 = adapter_sha256(adapter)
            if expected_adapter_sha256 != self.adapter_sha256:
                raise ValueError('adapter files differ from the expected SHA-256')
            self._model, self._tokenizer = load(str(path), adapter_path=str(adapter))
        if thinking not in ('server_default', 'off', 'on'):
            raise ValueError('unsupported thinking mode')
        template = getattr(self._tokenizer, 'chat_template', None)
        self.supports_thinking_switch = isinstance(template, str) and 'enable_thinking' in template
        if thinking in ('off', 'on') and not self.supports_thinking_switch:
            raise ValueError('tokenizer template does not expose thinking control')
        self.thinking = thinking
        self.thinking_canary = self._run_thinking_canary()
        if thinking in ('off', 'on') and not self.thinking_canary['different_token_ids']:
            raise ValueError('thinking on/off canary produced identical tokenizer input IDs')
        self._sampler = make_sampler(temp=temperature)
        self._seed = seed

    def _run_thinking_canary(self) -> dict[str, Any]:
        """Prove that the pinned tokenizer template changes IDs for on vs off."""
        if not self.supports_thinking_switch:
            return {'supported': False, 'different_token_ids': False,
                    'off_token_count': None, 'on_token_count': None,
                    'off_sha256': None, 'on_sha256': None}
        messages = [{'role': 'system', 'content': 'Return only the requested answer.'},
                    {'role': 'user', 'content': 'Reply exactly: OK'}]
        off = self.encode_messages(messages, thinking_mode='off')
        on = self.encode_messages(messages, thinking_mode='on')
        off_sha = hashlib.sha256(','.join(map(str, off)).encode()).hexdigest()
        on_sha = hashlib.sha256(','.join(map(str, on)).encode()).hexdigest()
        different = off != on
        if not different:
            self.supports_thinking_switch = False
        return {'supported': different, 'different_token_ids': different,
                'off_token_count': len(off), 'on_token_count': len(on),
                'off_sha256': off_sha, 'on_sha256': on_sha}

    def set_seed(self, seed: int) -> None:
        if type(seed) is not int:
            raise ValueError('seed must be an integer')
        self._seed = seed

    def encode_messages(self, messages: list[dict[str, str]], *, thinking_mode: str | None = None) -> list[int]:
        mode = self.thinking if thinking_mode is None else thinking_mode
        if mode not in ('server_default', 'off', 'on'):
            raise ValueError('unsupported thinking mode')
        kwargs = {'tokenize': True, 'add_generation_prompt': True, 'return_dict': False}
        if mode in ('off', 'on'):
            if not getattr(self, 'supports_thinking_switch', False):
                raise ValueError('tokenizer template does not expose thinking control')
            kwargs['enable_thinking'] = mode == 'on'
        ids = self._tokenizer.apply_chat_template(messages, **kwargs)
        if hasattr(ids, 'tolist'):
            ids = ids.tolist()
        if not isinstance(ids, list) or not ids or any(type(token) is not int or token < 0 for token in ids):
            raise ValueError('tokenizer returned invalid prompt token IDs')
        return ids

    @property
    def thinking_modes_supported(self) -> tuple[str, ...]:
        return ('server_default', 'off', 'on') if getattr(self, 'thinking_canary', {}).get(
            'different_token_ids', False) else ('server_default',)

    def generate_ids(self, prompt_ids: list[int], max_tokens: int) -> Iterator[int]:
        self._mx.random.seed(self._seed)
        for token, _ in self._generate_step(self._mx.array(prompt_ids), self._model,
                                             max_tokens=max_tokens, sampler=self._sampler):
            yield int(token)

    def decode(self, token_ids: list[int]) -> str:
        result: Any = self._tokenizer.decode(token_ids)
        if not isinstance(result, str):
            raise ValueError('tokenizer returned non-text output')
        return result

    @property
    def stop_token_ids(self) -> frozenset[int]:
        ids = getattr(self._tokenizer, 'eos_token_ids', None)
        if ids is None:
            eos = getattr(self._tokenizer, 'eos_token_id', None)
            ids = [] if eos is None else [eos]
        return frozenset(int(value) for value in ids)
