"""Opt-in fixture: random tiny Qwen-shaped model, no real Qwen weights or ARC data."""
import importlib.util
import json
import os
import platform
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(os.environ.get('RSI_RUN_TINY_MLX_FIXTURE') == '1' and platform.system() == 'Darwin'
                     and importlib.util.find_spec('mlx') is not None, 'explicit macOS MLX fixture required')
class TinyPipelineTests(unittest.TestCase):
    def test_restricted_tiny_training_and_reload(self):
        import mlx.core as mx
        from mlx.utils import tree_flatten
        from mlx_lm.models.qwen3 import Model, ModelArgs
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import PreTrainedTokenizerFast
        from src.arc.rsi_tiny_pipeline import run_tiny_pipeline_fixture

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root/'fabricated-model'
            model_dir.mkdir()
            config = {'model_type':'qwen3', 'hidden_size':16, 'num_hidden_layers':2, 'intermediate_size':32,
                      'num_attention_heads':2, 'rms_norm_eps':1e-6, 'vocab_size':12, 'num_key_value_heads':1,
                      'max_position_embeddings':128, 'rope_theta':10000.0, 'head_dim':8, 'tie_word_embeddings':True}
            (model_dir/'config.json').write_text(json.dumps(config))
            mx.random.seed(42)
            model = Model(ModelArgs.from_dict(config))
            mx.eval(model.parameters())
            mx.save_safetensors(str(model_dir/'model.safetensors'), dict(tree_flatten(model.parameters())))
            del model
            vocab = {w:i for i,w in enumerate(['[UNK]', '[PAD]', '[EOS]', 'user', 'assistant', 'hello',
                                                'world', 'answer', 'yes', 'no', ':', '.'])}
            backend = Tokenizer(WordLevel(vocab, unk_token='[UNK]'))
            backend.pre_tokenizer = Whitespace()
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='[UNK]',
                                                pad_token='[PAD]', eos_token='[EOS]')
            tokenizer.chat_template = "{% if enable_thinking %}{% endif %}{% for message in messages %}{{ message['role'] + ': ' + message['content'] + ' ' }}{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}"
            tokenizer.save_pretrained(model_dir)
            data = root/'synthetic'
            data.mkdir()
            record = {'messages':[{'role':'user','content':'hello world'}, {'role':'assistant','content':'answer yes'}]}
            (data/'train.jsonl').write_text(json.dumps(record)+'\n')
            result = run_tiny_pipeline_fixture(root=root, python_executable=os.sys.executable)
            self.assertEqual(result['stop_reason'], 'process_exited')
            self.assertEqual(result['exit_code'], 0)
            self.assertTrue(result['stream_complete'])
            self.assertEqual(result['loss']['last_step'], 1)
            self.assertTrue(result['tiny_adapter_reloaded'])
            self.assertEqual(result['tiny_reloaded_factor_matches'],28)
            self.assertEqual(result['tiny_reload_stop_reason'],'process_exited')
            self.assertGreater(result['tiny_reload_sample_count'],0)
            self.assertFalse(result['tiny_reload_cleanup_verified'])
            self.assertFalse(result['training_verified'])
            self.assertFalse(result['promotion_allowed'])
