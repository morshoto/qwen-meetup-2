"""Opt-in tiny-model trainer integration; never loads the pinned 14B weights."""
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch


@unittest.skipUnless(os.environ.get('LT_PHASE_B_MLX_PREFLIGHT') == '1', 'local MLX preflight not requested')
class TinyCanaryIntegration(unittest.TestCase):
    def test_installed_trainer_one_step_with_tiny_model(self):
        import mlx.core as mx
        from mlx.utils import tree_flatten
        from mlx_lm import lora
        from mlx_lm.models.qwen3 import Model, ModelArgs
        from mlx_lm.tuner.utils import linear_to_lora_layers
        from mlx_lm.utils import load_tokenizer
        from src.arc.lt_phase_b_canary_config import build_canary_config
        from src.arc.lt_phase_b_canary_trainer import run_instrumented_cli
        from src.arc.mlx_chat_template import install_training_template_policy
        from src.arc.rsi_loss_watch import TrainingLossWatch

        mx.set_memory_limit(2 * 1024**3)
        args = ModelArgs(model_type='qwen3', hidden_size=16, num_hidden_layers=8, intermediate_size=32,
            num_attention_heads=2, rms_norm_eps=1e-6, vocab_size=151936, num_key_value_heads=1,
            max_position_embeddings=512, rope_theta=10000., head_dim=8, tie_word_embeddings=True)
        model_path = Path.home() / '.cache/huggingface/hub/models--mlx-community--Qwen3-14B-4bit/snapshots/a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4'
        tokenizer = load_tokenizer(model_path)
        probe = Model(args); probe.freeze()
        linear_to_lora_layers(probe, 8, {'rank': 8, 'scale': 2.0, 'dropout': 0.0})
        shapes = {key: list(value.shape) for key, value in tree_flatten(probe.trainable_parameters())}
        self.assertTrue(shapes)
        del probe
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); data = root / 'data'; data.mkdir(mode=0o700)
            evidence = root / 'evidence'; evidence.mkdir(mode=0o700)
            plan = build_canary_config(model_dir=model_path, data_dir=data, adapter_dir=root / 'adapter')
            train = data / 'train.jsonl'; train.write_text(plan['dataset_jsonl']); train.chmod(0o600)
            config = root / 'config.json'; config.write_text(json.dumps(plan['trainer_config'], sort_keys=True)); config.chmod(0o600)
            config_sha = hashlib.sha256(config.read_bytes()).hexdigest()
            original_train = lora.train
            tiny = Model(args)
            base_file = root / 'tiny-base.safetensors'
            mx.save_safetensors(str(base_file), dict(tree_flatten(tiny.parameters())))
            output = io.StringIO()
            with patch.object(lora, 'load', return_value=(tiny, tokenizer)) as loader, \
                 patch('src.arc.lt_phase_b_tensor_evidence._expected_shapes', return_value=shapes), \
                 redirect_stdout(output):
                result = run_instrumented_cli(config_path=config, expected_config_sha256=config_sha,
                    evidence_dir=evidence, trainer_module=lora, install_policy=install_training_template_policy,
                    flatten=tree_flatten, evaluate=mx.eval)
            self.assertEqual(loader.call_count, 1)
            self.assertIs(lora.train, original_train)
            watch = TrainingLossWatch(); watch.feed(output.getvalue().encode()); watch.finish()
            self.assertEqual(watch.snapshot()['status'], 'ok')
            self.assertEqual(watch.snapshot()['last_step'], 1)
            self.assertGreater(result['changed_adapter_tensor_count'], 0)
            self.assertFalse(result['training_verified'])
            self.assertTrue((root / 'adapter/adapters.safetensors').is_file())
            self.assertTrue((root / 'adapter/adapter_config.json').is_file())
            from src.arc.lt_phase_b_adapter_artifact import inspect_adapter_artifact
            final_tensors = json.loads((evidence / 'tensor-after.json').read_text())['adapter']
            with patch('src.arc.lt_phase_b_adapter_artifact._expected_shapes', return_value=shapes):
                artifact = inspect_adapter_artifact(root / 'adapter', expected_tensors=final_tensors)
            self.assertEqual(artifact['matched_tensor_count'], len(shapes))
            self.assertFalse(artifact['reload_verified'])
            reload_script = """import json,sys
import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten
from mlx_lm.models.qwen3 import Model,ModelArgs
from mlx_lm.tuner.utils import load_adapters
a=ModelArgs(model_type='qwen3',hidden_size=16,num_hidden_layers=8,intermediate_size=32,num_attention_heads=2,rms_norm_eps=1e-6,vocab_size=151936,num_key_value_heads=1,max_position_embeddings=512,rope_theta=10000.,head_dim=8,tie_word_embeddings=True)
m=Model(a);m.load_weights(sys.argv[1],strict=True);m=load_adapters(m,sys.argv[2]);m.eval()
saved=mx.load(sys.argv[2]+'/adapters.safetensors')
loaded={k:v for k,v in tree_flatten(m.parameters()) if k.endswith(('.lora_a','.lora_b'))}
assert set(saved)==set(loaded) and saved
for k in saved:
 mx.eval(saved[k],loaded[k]);assert np.array_equal(np.asarray(saved[k]),np.asarray(loaded[k]))
logits=m(mx.array([[1,2]],dtype=mx.int32));mx.eval(logits)
print(json.dumps({'count':len(saved),'finite_logits':bool(mx.all(mx.isfinite(logits)).item())}))
"""
            child = subprocess.run([sys.executable, '-I', '-B', '-c', reload_script,
                                    str(base_file), str(root / 'adapter')], capture_output=True,
                                   text=True, timeout=60, check=True)
            reloaded = json.loads(child.stdout)
            self.assertEqual(reloaded['count'], len(shapes))
            self.assertTrue(reloaded['finite_logits'])


if __name__ == '__main__': unittest.main()
