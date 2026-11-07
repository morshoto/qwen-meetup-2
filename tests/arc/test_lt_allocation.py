import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.arc.lt_allocation import build_comparison_proposal, check_protected_allocation
from src.arc.lt_comparison import make_manifest
from src.arc.lt_development import build_development_manifest
from src.arc.rsi_exploratory_split import propose_exploratory_split


def raw(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


class AllocationTests(unittest.TestCase):
    def setUp(self):
        self.ids = [f't{i:04}' for i in range(1000)]
        self.known = set(self.ids[:100])
        self.challenges = raw({task_id: {} for task_id in self.ids})
        self.data_sha = hashlib.sha256(self.challenges).hexdigest()
        bounded = {'status': 'bounded_exposure_proposed_unverified', 'execution_ready': False,
                   'task_source_sha256': self.data_sha, 'known_exposed_task_ids': sorted(self.known),
                   'partitions': {'generation': self.ids[:800], 'development': self.ids[800:]}}
        self.registry = raw(bounded)
        self.registry_sha = hashlib.sha256(self.registry).hexdigest()
        selection = propose_exploratory_split(self.registry, expected_sha256=self.registry_sha)
        code = Path('src/arc/rsi_exploratory_split.py').read_bytes()
        selection['selection_code_sha256'] = hashlib.sha256(code).hexdigest()
        self.split = raw(selection)
        self.protected = set(selection['confirmatory_task_ids'] + selection['exploratory_task_ids'])
        self.development = build_development_manifest(self.ids, known_exposed=self.known,
            reserved=set(self.ids[800:]), dataset_sha256=self.data_sha,
            exposure_sha256=self.registry_sha, seed=20261111)
        self.training = {'t0100', 't0101'}
        self.proposal = make_manifest(self.ids, seed=2, development=set(self.development['task_ids']),
            training=self.training, exposures=self.known, protected=self.protected, dataset_sha256=self.data_sha)

    def test_builds_proposal_from_bound_inputs_without_freezing_or_execution(self):
        proposal, audit = build_comparison_proposal(self.challenges, self.registry, self.split,
                                                     self.development, sorted(self.training), seed=2)
        self.assertEqual(proposal, self.proposal)
        self.assertEqual(audit['protected_count'], 250)
        self.assertFalse(audit['execution_ready'])
        self.assertFalse(audit['training_inventory_independently_verified'])
        with self.assertRaises(ValueError):
            build_comparison_proposal(self.challenges, self.registry, self.split,
                                      self.development, [sorted(self.protected)[0]], seed=2)
        self.assertTrue(set(proposal['task_ids']).isdisjoint(self.protected | self.training | self.known))
        with self.assertRaises(ValueError):
            build_comparison_proposal(self.challenges + b' ', self.registry, self.split,
                                      self.development, sorted(self.training), seed=2)
        with self.assertRaises(ValueError):
            build_comparison_proposal(self.challenges, self.registry, self.split,
                                      self.development, [self.development['task_ids'][0]], seed=2)

    def test_cli_requires_training_inventory_and_checks_proposal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            documents = {'challenges': self.challenges, 'registry': self.registry, 'split': self.split,
                         'development': raw(self.development), 'training': raw(sorted(self.training))}
            for name, value in documents.items(): (root / name).write_bytes(value)
            command = [sys.executable, '-m', 'src.arc.lt_allocation']
            for name in documents: command += ['--' + name, str(root / name)]
            command += ['--seed', '2']
            proposal = subprocess.run(command + ['--part', 'proposal'], capture_output=True, text=True)
            audit = subprocess.run(command + ['--part', 'audit'], capture_output=True, text=True)
            self.assertEqual(proposal.returncode, 0, proposal.stderr)
            self.assertEqual(audit.returncode, 0, audit.stderr)
            self.assertEqual(json.loads(proposal.stdout), self.proposal)
            self.assertFalse(json.loads(audit.stdout)['execution_ready'])
            (root / 'training').write_bytes(raw([self.development['task_ids'][0]]))
            failed = subprocess.run(command + ['--part', 'proposal'], capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertEqual(failed.stdout, '')

    def test_checks_prior_reservations_development_and_training(self):
        result = check_protected_allocation(self.proposal, self.registry, self.split,
                                            self.development, sorted(self.training))
        self.assertEqual(result['protected_count'], 250)
        self.assertFalse(result['execution_ready'])
        self.assertFalse(result['training_inventory_independently_verified'])
        self.assertTrue(set(self.proposal['task_ids']).isdisjoint(self.protected))

    def test_rejects_omitted_protection_or_training_and_tampered_split(self):
        proposal = make_manifest(self.ids, seed=2, development=set(self.development['task_ids']),
            training=self.training, exposures=self.known, protected=set(), dataset_sha256=self.data_sha)
        with self.assertRaises(ValueError):
            check_protected_allocation(proposal, self.registry, self.split, self.development, sorted(self.training))
        with self.assertRaises(ValueError):
            check_protected_allocation(self.proposal, self.registry, self.split, self.development, ['t0102'])
        changed = json.loads(self.split)
        changed['exploratory_task_ids'][0] = 't0000'
        with self.assertRaises(ValueError):
            check_protected_allocation(self.proposal, self.registry, raw(changed), self.development, sorted(self.training))


if __name__ == '__main__': unittest.main()
