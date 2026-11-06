import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.arc.lt_allocation import build_comparison_proposal
from src.arc.lt_development import build_development_manifest
from src.arc.lt_source import capture_source_tree
from src.arc.rsi_exploratory_split import propose_exploratory_split


def raw(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


class RegisteredPreparationTests(unittest.TestCase):
    def test_cli_requires_and_rechecks_protected_allocation_and_source_tree(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ids = [f't{i:04}' for i in range(1000)]
            challenges = {task_id: {'train': [{'input': [[1]], 'output': [[1]]}],
                                    'test': [{'input': [[2]]}]} for task_id in ids}
            challenge_bytes = raw(challenges)
            data_sha = hashlib.sha256(challenge_bytes).hexdigest()
            known = set(ids[:100])
            registry_bytes = raw({'status': 'bounded_exposure_proposed_unverified', 'execution_ready': False,
                'task_source_sha256': data_sha, 'known_exposed_task_ids': sorted(known),
                'partitions': {'generation': ids[:800], 'development': ids[800:]}})
            registry_sha = hashlib.sha256(registry_bytes).hexdigest()
            split = propose_exploratory_split(registry_bytes, expected_sha256=registry_sha)
            split['selection_code_sha256'] = hashlib.sha256(Path('src/arc/rsi_exploratory_split.py').read_bytes()).hexdigest()
            split_bytes = raw(split)
            development = build_development_manifest(ids, known_exposed=known, reserved=set(ids[800:]),
                dataset_sha256=data_sha, exposure_sha256=registry_sha, seed=20261111)
            training = ['t0100', 't0101']
            proposal, _ = build_comparison_proposal(challenge_bytes, registry_bytes, split_bytes,
                                                      development, training, seed=2)
            tree = root / 'tree'
            tree.mkdir()
            for name in ('src', 'sandbox', 'configs', 'scripts'):
                (tree / name).mkdir()
                (tree / name / 'fixture').write_text(name)
            for name in ('pyproject.toml', 'uv.lock'): (tree / name).write_text(name)
            inventory = capture_source_tree(tree)
            sources = {name: str(tree / name) for name in inventory['files']}
            prompt = b'fixture prompt'
            config = {'prompt_sha256': hashlib.sha256(prompt).hexdigest(), 'code_sha256': inventory['code_sha256'],
                'models': {'14b': '1' * 40, '32b': '2' * 40}, 'model_names': {'14b': 'model14', '32b': 'model32'},
                'backend_url': 'http://127.0.0.1:8080/v1', 'thinking': 'off', 'temperature': .7,
                'candidates': 2, 'max_tokens': 1000, 'max_prompt_tokens': 5000, 'timeout_seconds': 60,
                'seed': 42, 'max_repairs': 1, 'prompt_variant': 'legacy',
                'model_files_sha256': {'14b': 'd' * 64, '32b': 'e' * 64}, 'runtime_sha256': 'f' * 64,
                'sandbox': {'image': 'sha256:' + '9' * 64, 'timeout': 3, 'memory_mb': 128, 'cpus': 1, 'pids': 32},
                'budget': {'max_calls': 2000, 'max_reserved_tokens': 20000000, 'max_sandbox_starts': 8000,
                    'max_seconds': 36000, 'max_calls_per_task': 10, 'max_reserved_tokens_per_task': 100000,
                    'max_sandbox_per_task': 40, 'max_seconds_per_task': 120}}
            documents = {'proposal': raw(proposal), 'config': raw(config), 'challenges': challenge_bytes,
                'solutions': raw({task_id: [[[2]]] for task_id in ids}), 'sources': raw(sources), 'prompt': prompt,
                'registry': registry_bytes, 'split': split_bytes, 'development': raw(development),
                'training': raw(training), 'source-inventory': raw(inventory)}
            for name, value in documents.items(): (root / name).write_bytes(value)
            base = [sys.executable, '-m', 'src.arc.lt_prepare_cli']
            for name in documents: base += ['--' + name, str(root / name)]
            base += ['--source-root', str(tree)]
            frozen = subprocess.run(base + ['--part', 'frozen'], capture_output=True, text=True)
            receipt = subprocess.run(base + ['--part', 'receipt'], capture_output=True, text=True)
            self.assertEqual(frozen.returncode, 0, frozen.stderr)
            self.assertEqual(receipt.returncode, 0, receipt.stderr)
            saved = json.loads(receipt.stdout)
            self.assertEqual(saved['allocation_scope'], 'prior_reservations_checked_unverified')
            self.assertEqual(saved['source_scope'], 'conservative_source_tree')
            (root / 'frozen').write_text(frozen.stdout)
            (root / 'receipt').write_text(receipt.stdout)
            (root / 'results').write_text('')
            score = [sys.executable, '-m', 'src.arc.lt_score_cli', '--frozen', str(root / 'frozen'),
                     '--receipt', str(root / 'receipt'), '--results', str(root / 'results'), '--source-root', str(tree)]
            for name in ('challenges', 'solutions', 'sources', 'prompt', 'registry', 'split',
                         'development', 'training', 'source-inventory'):
                score += ['--' + name, str(root / name)]
            result = subprocess.run(score, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(json.loads(result.stdout)['evidence']['run_identity_independently_verified'])
            (root / 'training').write_bytes(raw(['t0100']))
            failed = subprocess.run(score, capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertEqual(failed.stdout, '')


if __name__ == '__main__': unittest.main()
