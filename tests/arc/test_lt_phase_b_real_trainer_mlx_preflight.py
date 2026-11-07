"""Opt-in trainer integration with synthetic fixture tasks and a tiny model."""
import hashlib
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch


@unittest.skipUnless(os.environ.get('LT_PHASE_B_MLX_PREFLIGHT') == '1', 'local MLX preflight not requested')
class TinyRealTrainerIntegration(unittest.TestCase):
    def test_fixed_real_data_path_with_installed_trainer_and_tiny_model(self):
        import mlx.core as mx
        from mlx.utils import tree_flatten
        from mlx_lm import lora
        from mlx_lm.models.qwen3 import Model, ModelArgs
        from mlx_lm.tuner.datasets import ChatDataset
        from mlx_lm.tuner.utils import linear_to_lora_layers
        from mlx_lm.utils import load_tokenizer
        from src.arc.lt_phase_b_sft_materialize import materialize_sft
        from src.arc.lt_phase_b_training_config import build_training_config
        from src.arc.lt_phase_b_training_trainer import run_real_instrumented_cli
        from src.arc.mlx_chat_template import install_training_template_policy
        from src.arc.rsi_loss_watch import TrainingLossWatch, _REPORT
        from tests.arc import test_lt_phase_b_sft_preflight as fixtures

        mx.set_memory_limit(2 * 1024**3)
        args = ModelArgs(model_type='qwen3', hidden_size=16, num_hidden_layers=8, intermediate_size=32,
            num_attention_heads=2, rms_norm_eps=1e-6, vocab_size=151936, num_key_value_heads=1,
            max_position_embeddings=512, rope_theta=10000., head_dim=8, tie_word_embeddings=True)
        model_path = Path.home() / '.cache/huggingface/hub/models--mlx-community--Qwen3-14B-4bit/snapshots/a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4'
        tokenizer = load_tokenizer(model_path)
        install_training_template_policy('off')
        fixture = fixtures.SFTPreflightTests('test_audits_exact_verified_examples_and_prefix'); fixture.setUp()
        probe = Model(args); probe.freeze()
        linear_to_lora_layers(probe, 8, {'rank': 8, 'scale': 2.0, 'dropout': 0.0})
        shapes = {key: list(value.shape) for key, value in tree_flatten(probe.trainable_parameters())}
        del probe
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / 'artifacts').mkdir()
            run = root / 'artifacts/lt-phase-b-fixture-run'; run.mkdir(mode=0o700)
            materialize_sft(source_root=root, run_dir=run, dataset=fixture.dataset,
                tasks=fixture.fixture.tasks, verifier=fixture.fixture.verifier, source=fixture.fixture.source,
                tokenizer=tokenizer, dataset_factory=lambda example, tok: ChatDataset([example], tok, mask_prompt=True))
            plan = build_training_config(source_root=root, run_dir=run, model_dir=model_path,
                                         expected_source=fixture.fixture.source)
            config = run / 'training-config.json'; config.write_text(json.dumps(plan['trainer_config'], sort_keys=True)); config.chmod(0o600)
            digest = hashlib.sha256(config.read_bytes()).hexdigest()
            evidence = run / 'training-evidence'; evidence.mkdir(mode=0o700)
            output = io.StringIO()
            with patch.object(lora, 'load', return_value=(Model(args), tokenizer)), \
                 patch('src.arc.lt_phase_b_tensor_evidence._expected_shapes', return_value=shapes), redirect_stdout(output):
                result = run_real_instrumented_cli(source_root=root, run_dir=run, model_dir=model_path,
                    expected_source=fixture.fixture.source, config_path=config, expected_config_sha256=digest,
                    evidence_dir=evidence, trainer_module=lora, install_policy=install_training_template_policy,
                    flatten=tree_flatten, evaluate=mx.eval)
            watch = TrainingLossWatch(); watch.feed(output.getvalue().encode()); watch.finish()
            self.assertEqual(watch.snapshot()['status'], 'ok')
            self.assertEqual(watch.snapshot()['last_step'], 20)
            reports = [int(match.group(1)) for line in output.getvalue().splitlines() if (match := _REPORT.match(line))]
            self.assertEqual(reports, list(range(1, 21)))
            self.assertGreater(result['changed_adapter_tensor_count'], 0)
            self.assertFalse(result['training_verified'])
            self.assertTrue((run / 'gen1-adapter/adapters.safetensors').exists())


if __name__ == '__main__': unittest.main()
