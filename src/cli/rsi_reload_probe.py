"""Reload one registered model adapter; supervision and authorization are external.

This entrypoint must be invoked only inside an independently monitored process.
It does not authorize execution, attest the running process, or promote adapters.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

from src.arc.rsi_adapter_reload import observe_model_reload
from src.arc.rsi_transport_audit import _json_object


def _runtime():
    import mlx.core as mx
    import numpy as np
    from mlx.utils import tree_flatten
    from mlx_lm.utils import load
    return load, tree_flatten, mx.eval, np.asarray


def main(argv=None):
    parser=argparse.ArgumentParser(description='Observe a quarantined adapter reload under external supervision')
    parser.add_argument('--model-key',choices=('14b','32b'),required=True)
    parser.add_argument('--model-dir',required=True)
    parser.add_argument('--adapter-dir',required=True)
    parser.add_argument('--adapter-sha256',required=True)
    parser.add_argument('--model-manifest',type=Path,required=True)
    parser.add_argument('--model-manifest-sha256',required=True)
    args=parser.parse_args(argv)
    raw=args.model_manifest.read_bytes()
    if not 1 <= len(raw) <= 8*1024*1024 or hashlib.sha256(raw).hexdigest()!=args.model_manifest_sha256:
        raise ValueError('reload model manifest digest differs')
    manifest=_json_object(raw)
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['TRANSFORMERS_OFFLINE']='1'
    load,flatten,evaluate,to_numpy=_runtime()
    result=observe_model_reload(model_key=args.model_key,model_dir=args.model_dir,adapter_dir=args.adapter_dir,
        expected_adapter_sha256=args.adapter_sha256,model_manifest=manifest,load_model=load,
        flatten_parameters=flatten,evaluate=evaluate,to_numpy=to_numpy)
    print(json.dumps(result,sort_keys=True))


if __name__ == '__main__':
    main()
