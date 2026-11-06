import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src.arc.lt_attestation import source_digest, attest_inputs, verify_inputs
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
                  'models': {'14b': 'revision-14', '32b': 'revision-32'},
                  'thinking': 'off', 'temperature': 0.7, 'candidates': 2,
                  'max_tokens': 1000, 'timeout_seconds': 60}
        self.frozen = freeze_manifest(proposal, config)

    def test_attestation_binds_data_sources_and_prompt(self):
        receipt = attest_inputs(self.frozen, self.challenge, self.solution, self.sources, self.prompt)
        self.assertEqual(len(receipt['selected_task_sha256']), 200)
        verify_inputs(receipt, self.frozen, self.challenge, self.solution, self.sources, self.prompt)
        self.solution.write_text(self.solution.read_text() + ' ')
        with self.assertRaises(ValueError):
            verify_inputs(receipt, self.frozen, self.challenge, self.solution, self.sources, self.prompt)

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
