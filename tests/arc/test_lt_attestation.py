import hashlib
import json
import tempfile
import subprocess
import sys
import unittest
from pathlib import Path

from src.arc.lt_attestation import source_digest, attest_inputs, verify_inputs, attest_model_runtime, verify_model_runtime
from src.arc.lt_comparison import make_manifest, freeze_manifest


class InputAttestationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.challenge = root / 'challenges.json'
        self.solution = root / 'solutions.json'
        self.code = root / 'solver.py'
        self.prompt = root / 'prompt.txt'
        challenges = {f't{i:03}': {'train': [{'input': [[1]], 'output': [[1]]}],
                                     'test': [{'input': [[2]]}]} for i in range(210)}
        solutions = {key: [[[2]]] for key in challenges}
        self.challenge.write_text(json.dumps(challenges))
        self.solution.write_text(json.dumps(solutions))
        self.code.write_text('solver source')
        self.prompt.write_text('prompt template')
        self.sources = {'solver.py': self.code}
        digest = hashlib.sha256(self.challenge.read_bytes()).hexdigest()
        proposal = make_manifest(list(challenges), seed=7, development={'t000'}, training={'t001'},
                                 exposures={'t002'}, dataset_sha256=digest)
        config = {'prompt_sha256': hashlib.sha256(self.prompt.read_bytes()).hexdigest(),
                  'code_sha256': source_digest(self.sources),
                  'models': {'14b': '1' * 40, '32b': '2' * 40},
                  'model_names': {'14b': 'model14', '32b': 'model32'}, 'backend_url': 'http://127.0.0.1:8080/v1',
                  'sandbox': {'image': 'sha256:' + '9' * 64, 'timeout': 3, 'memory_mb': 128, 'cpus': 1, 'pids': 32},
                  'thinking': 'off', 'temperature': 0.7, 'candidates': 2,
                  'max_tokens': 1000, 'max_prompt_tokens': 5000, 'timeout_seconds': 60, 'seed': 42, 'max_repairs': 1,
                  'prompt_variant': 'legacy', 'model_files_sha256': {'14b': 'd' * 64, '32b': 'e' * 64},
                  'runtime_sha256': 'f' * 64, 'budget': {'max_calls': 2000,
                  'max_reserved_tokens': 20000000, 'max_sandbox_starts': 8000,
                  'max_seconds': 36000, 'max_calls_per_task': 10,
                  'max_reserved_tokens_per_task': 100000, 'max_sandbox_per_task': 40,
                  'max_seconds_per_task': 120}}
        self.frozen = freeze_manifest(proposal, config)

    def test_attestation_binds_data_sources_and_prompt(self):
        receipt = attest_inputs(self.frozen, self.challenge, self.solution, self.sources, self.prompt)
        self.assertEqual(len(receipt['selected_task_sha256']), 200)
        verify_inputs(receipt, self.frozen, self.challenge, self.solution, self.sources, self.prompt)
        self.solution.write_text(self.solution.read_text() + ' ')
        with self.assertRaises(ValueError):
            verify_inputs(receipt, self.frozen, self.challenge, self.solution, self.sources, self.prompt)

    def test_full_source_attestation_rejects_added_source_after_freeze(self):
        from src.arc.lt_source import capture_source_tree
        root = self.challenge.parent / 'tree'
        root.mkdir()
        for name in ('src', 'sandbox', 'configs', 'scripts'):
            (root / name).mkdir()
            (root / name / 'file').write_text(name)
        for name in ('pyproject.toml', 'uv.lock'):
            (root / name).write_text(name)
        inventory = capture_source_tree(root)
        sources = {name: root / name for name in inventory['files']}
        config = dict(self.frozen['config'])
        config['code_sha256'] = inventory['code_sha256']
        frozen = freeze_manifest(self.frozen['manifest'], config)
        receipt = attest_inputs(frozen, self.challenge, self.solution, sources, self.prompt,
                                source_root=root, source_inventory=inventory)
        verify_inputs(receipt, frozen, self.challenge, self.solution, sources, self.prompt,
                      source_root=root, source_inventory=inventory)
        workspace = self.challenge.parent
        (workspace / 'frozen.json').write_text(json.dumps(frozen))
        (workspace / 'receipt.json').write_text(json.dumps(receipt))
        (workspace / 'inventory.json').write_text(json.dumps(inventory))
        (workspace / 'sources.json').write_text(json.dumps({name: str(path) for name, path in sources.items()}))
        (workspace / 'results.jsonl').write_text('')
        (workspace / 'proposal.json').write_text(json.dumps(frozen['manifest']))
        (workspace / 'config.json').write_text(json.dumps(frozen['config']))
        prepare = [sys.executable, '-m', 'src.arc.lt_prepare_cli', '--proposal', str(workspace / 'proposal.json'),
                   '--config', str(workspace / 'config.json'), '--challenges', str(self.challenge),
                   '--solutions', str(self.solution), '--sources', str(workspace / 'sources.json'),
                   '--prompt', str(self.prompt), '--part', 'receipt', '--source-root', str(root),
                   '--source-inventory', str(workspace / 'inventory.json')]
        prepared = subprocess.run(prepare, capture_output=True, text=True)
        self.assertEqual(prepared.returncode, 0, prepared.stderr)
        self.assertEqual(json.loads(prepared.stdout), receipt)
        command = [sys.executable, '-m', 'src.arc.lt_score_cli', '--frozen', str(workspace / 'frozen.json'),
                   '--receipt', str(workspace / 'receipt.json'), '--challenges', str(self.challenge),
                   '--solutions', str(self.solution), '--sources', str(workspace / 'sources.json'),
                   '--prompt', str(self.prompt), '--results', str(workspace / 'results.jsonl'),
                   '--source-root', str(root), '--source-inventory', str(workspace / 'inventory.json')]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        (root / 'src' / 'added.py').write_text('changed source closure')
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, '')
        with self.assertRaises(ValueError):
            verify_inputs(receipt, frozen, self.challenge, self.solution, sources, self.prompt,
                          source_root=root, source_inventory=inventory)

    def test_model_and_runtime_files_must_match_registered_hashes(self):
        model14 = self.challenge.parent / 'model14'
        model32 = self.challenge.parent / 'model32'
        runtime = self.challenge.parent / 'runtime'
        model14.write_bytes(b'14b weights and tokenizer fixture')
        model32.write_bytes(b'32b weights and tokenizer fixture')
        runtime.write_bytes(b'local runtime fixture')
        config = dict(self.frozen['config'])
        config['model_files_sha256'] = {'14b': source_digest({'model': model14}),
                                        '32b': source_digest({'model': model32})}
        config['runtime_sha256'] = source_digest({'runtime': runtime})
        frozen = freeze_manifest(self.frozen['manifest'], config)
        receipt = attest_model_runtime(frozen, '14b', {'model': model14}, {'runtime': runtime})
        verify_model_runtime(receipt, frozen, '14b', {'model': model14}, {'runtime': runtime})
        model14.write_bytes(b'changed weights')
        with self.assertRaises(ValueError):
            verify_model_runtime(receipt, frozen, '14b', {'model': model14}, {'runtime': runtime})
        with self.assertRaises(ValueError):
            attest_model_runtime(frozen, '32b', {'model': model14}, {'runtime': runtime})

    def test_public_export_cli_only_emits_selected_challenges(self):
        root = self.challenge.parent
        receipt = attest_inputs(self.frozen, self.challenge, self.solution, self.sources, self.prompt)
        (root / 'frozen.json').write_text(json.dumps(self.frozen))
        (root / 'receipt.json').write_text(json.dumps(receipt))
        command = [sys.executable, '-m', 'src.arc.lt_public_cli', '--frozen', str(root / 'frozen.json'),
                   '--receipt', str(root / 'receipt.json'), '--challenges', str(self.challenge)]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        exported = json.loads(result.stdout)
        self.assertEqual(set(exported['tasks']), set(self.frozen['manifest']['task_ids']))
        self.assertTrue(all(set(pair) == {'input'} for task in exported['tasks'].values() for pair in task['test']))
        self.challenge.write_text(self.challenge.read_text() + ' ')
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, '')

    def test_offline_prepare_cli_emits_separate_reviewable_documents(self):
        root = self.challenge.parent
        (root / 'proposal.json').write_text(json.dumps(self.frozen['manifest']))
        (root / 'config.json').write_text(json.dumps(self.frozen['config']))
        (root / 'sources.json').write_text(json.dumps({'solver.py': str(self.code)}))
        command = [sys.executable, '-m', 'src.arc.lt_prepare_cli',
                   '--proposal', str(root / 'proposal.json'), '--config', str(root / 'config.json'),
                   '--challenges', str(self.challenge), '--solutions', str(self.solution),
                   '--sources', str(root / 'sources.json'), '--prompt', str(self.prompt)]
        frozen = subprocess.run(command + ['--part', 'frozen'], capture_output=True, text=True)
        receipt = subprocess.run(command + ['--part', 'receipt'], capture_output=True, text=True)
        self.assertEqual(frozen.returncode, 0, frozen.stderr)
        self.assertEqual(receipt.returncode, 0, receipt.stderr)
        self.assertEqual(json.loads(frozen.stdout), self.frozen)
        self.assertEqual(json.loads(receipt.stdout)['comparison_sha256'], self.frozen['sha256'])
        self.code.write_text('changed')
        failure = subprocess.run(command + ['--part', 'frozen'], capture_output=True, text=True)
        self.assertNotEqual(failure.returncode, 0)
        self.assertEqual(failure.stdout, '')

    def test_offline_cli_scores_bound_predictions_and_rejects_changed_answers(self):
        root = self.challenge.parent
        receipt = attest_inputs(self.frozen, self.challenge, self.solution, self.sources, self.prompt)
        (root / 'frozen.json').write_text(json.dumps(self.frozen))
        (root / 'receipt.json').write_text(json.dumps(receipt))
        (root / 'sources.json').write_text(json.dumps({'solver.py': str(self.code)}))
        task_id = self.frozen['manifest']['task_ids'][0]
        row = {'comparison_sha256': self.frozen['sha256'], 'model': '14b', 'task_id': task_id,
               'status': 'complete', 'candidates': [[[[2]]], [[[0]]]], 'tokens': 2,
               'runtime_seconds': 1, 'peak_memory_bytes': 10, 'peak_swap_bytes': 0}
        (root / 'results.jsonl').write_text(json.dumps(row) + '\n')
        command = [sys.executable, '-m', 'src.arc.lt_score_cli', '--frozen', str(root / 'frozen.json'),
                   '--receipt', str(root / 'receipt.json'), '--challenges', str(self.challenge),
                   '--solutions', str(self.solution), '--sources', str(root / 'sources.json'),
                   '--prompt', str(self.prompt), '--results', str(root / 'results.jsonl')]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report['models']['14b']['pass_at_1'], 1)
        self.assertEqual(report['models']['14b']['status'], 'incomplete')
        self.assertEqual(report['models']['32b']['attempted'], 0)
        result = subprocess.run(command + ['--format', 'jsonl'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = [json.loads(line) for line in result.stdout.splitlines()]
        self.assertEqual(len(lines), 401)
        self.assertEqual(lines[0]['record_type'], 'task')
        self.assertEqual(lines[0]['pass_at_1'], True)
        self.assertEqual(lines[-1]['record_type'], 'summary')
        self.assertEqual(lines[-1]['models']['14b']['status'], 'incomplete')
        self.solution.write_text(self.solution.read_text() + ' ')
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, '')

    def test_rejects_rehashed_but_nonrandom_task_selection(self):
        proposal = dict(self.frozen['manifest'])
        selected = list(proposal['task_ids'])
        replacement = next(x for x in (f't{i:03}' for i in range(3, 210)) if x not in selected)
        selected[0] = replacement
        proposal['task_ids'] = sorted(selected)
        rehashed = freeze_manifest(proposal, self.frozen['config'])
        with self.assertRaises(ValueError):
            attest_inputs(rehashed, self.challenge, self.solution, self.sources, self.prompt)

    def test_rejects_changed_code_prompt_or_task_selection(self):
        self.code.write_text('changed')
        with self.assertRaises(ValueError):
            attest_inputs(self.frozen, self.challenge, self.solution, self.sources, self.prompt)
        self.code.write_text('solver source')
        self.prompt.write_text('changed')
        with self.assertRaises(ValueError):
            attest_inputs(self.frozen, self.challenge, self.solution, self.sources, self.prompt)
        self.prompt.write_text('prompt template')
        self.frozen['manifest']['task_ids'][0] = 't000'
        with self.assertRaises(ValueError):
            attest_inputs(self.frozen, self.challenge, self.solution, self.sources, self.prompt)


if __name__ == '__main__': unittest.main()
