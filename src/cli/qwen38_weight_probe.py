"""Independently compare saved LoRA tensors and finite reloaded model logits."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

from src.arc.qwen38_local_runner import MODEL_REVISION


def max_logit_delta(base, adapted):
    first, second = np.asarray(base, dtype=np.float64), np.asarray(adapted, dtype=np.float64)
    if first.shape != second.shape or first.size == 0 or not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
        raise ValueError('nonfinite or mismatched probe logits')
    value = float(np.max(np.abs(first - second)))
    if not math.isfinite(value) or value <= 0:
        raise ValueError('adapter did not change probe logits')
    return value


def _sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def _write(path, value):
    data = (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(data); stream.flush(); os.fsync(stream.fileno())


def probe(model_path, run_root):
    model_path, root = Path(model_path).resolve(), Path(run_root).resolve()
    if not model_path.is_dir() or model_path.name != MODEL_REVISION or not root.is_dir() or root.stat().st_mode & 0o077:
        raise ValueError('invalid model or private probe directory')
    import mlx.core as mx
    from mlx_vlm import load
    initial = mx.load(str(root / 'initial.safetensors'))
    adapter_path = root / 'adapter/adapters.safetensors'
    final = mx.load(str(adapter_path))
    if not initial or set(initial) != set(final):
        raise ValueError('adapter tensor inventory changed')
    changed = 0; nonzero = 0
    for key in initial:
        if 'lora' not in key.lower() or initial[key].shape != final[key].shape:
            raise ValueError('unexpected trainable tensor')
        a, b = initial[key].astype(mx.float32), final[key].astype(mx.float32)
        finite = mx.all(mx.isfinite(a)) & mx.all(mx.isfinite(b))
        delta, magnitude = mx.max(mx.abs(a - b)), mx.max(mx.abs(b))
        mx.eval(finite, delta, magnitude)
        if not bool(finite.item()) or not math.isfinite(float(delta.item())):
            raise ValueError('nonfinite adapter tensor')
        changed += int(float(delta.item()) > 0); nonzero += int(float(magnitude.item()) > 0)
    if not changed or not nonzero:
        raise ValueError('adapter tensors did not change')
    del initial, final
    gc.collect(); mx.clear_cache()
    logits = []
    for adapter in (None, str(root / 'adapter')):
        model, processor = load(str(model_path), adapter_path=adapter)
        tokenizer = getattr(processor, 'tokenizer', processor)
        ids = tokenizer.encode('The next number after one is', add_special_tokens=False)
        if not ids: raise ValueError('empty probe token sequence')
        result = model.language_model(mx.array([ids]))
        value = result.logits if hasattr(result, 'logits') else result
        value = value[0, -1, :].astype(mx.float32); mx.eval(value)
        logits.append(np.asarray(value).copy())
        del model, processor, result, value
        gc.collect(); mx.clear_cache()
    delta = max_logit_delta(*logits)
    base = {'adapter_sha256': _sha(adapter_path), 'nonzero_lora_tensors': nonzero, 'max_abs_logit_delta': delta}
    _write(root / 'adapter/probe.json', base)
    result = {**base, 'changed_lora_tensors': changed}
    _write(root / 'tensor-probe.json', result)
    print(json.dumps(result, sort_keys=True), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True); parser.add_argument('--run-dir', required=True)
    args = parser.parse_args(argv)
    probe(args.model_path, args.run_dir)


if __name__ == '__main__':
    main()
