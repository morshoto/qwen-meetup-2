"""Opt-in real-tokenizer SFT preparation on synthetic fixture tasks only."""
import os
import unittest
from pathlib import Path
from unittest.mock import patch


@unittest.skipUnless(os.environ.get('LT_PHASE_B_TRAINER_PREFLIGHT') == '1', 'local MLX-LM preflight not requested')
class PrepareMLXPreflightTests(unittest.TestCase):
    def test_registered_preparation_uses_installed_chat_dataset(self):
        from src.arc.lt_phase_b_prepare_dispatch import _tokenizer_runtime
        from tests.arc import test_lt_phase_b_prepare_dispatch as fixtures
        fixture = fixtures.PrepareDispatchTests('test_replays_and_prepares_with_verified_tokenizer_runtime')
        fixture.setUp(); self.addCleanup(fixture.doCleanups)
        path = Path.home() / '.cache/huggingface/hub/models--mlx-community--Qwen3-14B-4bit/snapshots/a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4'
        with patch('src.arc.lt_phase_b_prepare_dispatch._tokenizer_runtime', side_effect=_tokenizer_runtime):
            result = fixture.invoke(model_path=path)
        self.assertEqual(result['status'], 'private_sft_prepared_unverified')
        self.assertEqual((result['train_count'], result['valid_count']), (2, 1))
        self.assertFalse(result['training_allowed'])


if __name__ == '__main__': unittest.main()
