"""Capture initialized trainable tensors immediately before the pinned trainer runs."""
from __future__ import annotations

import importlib.util
import os
import random
import sys
from pathlib import Path

SITE = '    model = setup_model_for_training(model, args, args.adapter_path)\n'
CAPTURE = '''    import mlx.core as _canary_mx
    from mlx.utils import tree_flatten as _canary_flatten
    _canary_mx.save_safetensors(os.environ['QWEN38_CANARY_INITIAL'], dict(_canary_flatten(model.trainable_parameters())))
'''


def instrument(source):
    if source.count(SITE) != 1:
        raise ValueError('unrecognized trainer initialization site')
    return source.replace(SITE, SITE + CAPTURE)


def main():
    import mlx.core as mx
    import numpy as np
    random.seed(20261111); np.random.seed(20261111); mx.random.seed(20261111)
    path = Path(importlib.util.find_spec('mlx_vlm.lora').origin)
    source = instrument(path.read_text())
    sys.argv[0] = str(path)
    namespace = {'__name__': '__main__', '__file__': str(path), '__package__': 'mlx_vlm', 'os': os}
    exec(compile(source, str(path), 'exec'), namespace)


if __name__ == '__main__':
    main()
