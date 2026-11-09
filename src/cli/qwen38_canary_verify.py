"""Compare saved LoRA tensors and probe the reloaded adapter's logits."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--run-dir', required=True)
    args = parser.parse_args()
    root = Path(args.run_dir)
    import mlx.core as mx
    from src.cli.qwen38_local_train import verify_adapter
    initial = mx.load(str(root / 'initial.safetensors'))
    final = mx.load(str(root / 'adapter/adapters.safetensors'))
    if set(initial) != set(final):
        raise ValueError('initial and final tensor sets differ')
    changes = 0
    for key in initial:
        if 'lora' not in key.lower():
            raise ValueError('unexpected trainable base tensor')
        if initial[key].shape != final[key].shape:
            raise ValueError('tensor shape mismatch')
        delta = mx.max(mx.abs(initial[key].astype(mx.float32) - final[key].astype(mx.float32)))
        mx.eval(delta)
        changes += int(float(delta.item()) > 0)
    del initial, final
    mx.clear_cache()
    if not changes:
        raise ValueError('no LoRA tensor changed')
    output = root / 'adapter/probe.json'
    if output.exists():
        raise ValueError('probe already exists')
    result = verify_adapter(Path(args.model_path), root / 'adapter', output)
    result['changed_lora_tensors'] = changes
    (root / 'tensor-probe.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
