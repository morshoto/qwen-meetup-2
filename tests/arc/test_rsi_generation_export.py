import hashlib
import json
import tempfile
import unittest
from unittest.mock import Mock

from tests.arc.test_rsi_allocation_review import AllocationReviewTests
from src.arc.rsi_generation_export import prepare_generation_export, load_generation_export
from src.arc.e6_curriculum import _quality_grids
from src.arc.e6_fake import FAKE_PROGRAMS, FakeE6Sandbox
from src.arc.rsi_collection_budget import CollectionBudget
from src.arc.rsi_generation_stage import collect_generation
from src.arc.rsi_lineage_audit import probes_sha256
from src.arc.rsi_slot_audit import task_sha256


class GenerationExportTests(AllocationReviewTests):
    def export(self):
        return prepare_generation_export(self.dataset, self.raw, self.selected,
                                         expected_bounded_sha256=self.sha)

    def test_export_contains_only_generation_tasks_and_no_test_answers(self):
        raw, receipt = self.export()
        exported = json.loads(raw)
        self.assertEqual(set(exported), set(self.selected['generation_task_ids']))
        self.assertFalse(set(exported) & set(self.selected['exploratory_task_ids']))
        self.assertFalse(set(exported) & set(self.selected['confirmatory_task_ids']))
        self.assertTrue(all(set(pair) == {'input'} for item in exported.values() for pair in item['test']))
        self.assertEqual(receipt['export_sha256'], hashlib.sha256(raw).hexdigest())
        self.assertFalse(receipt['execution_ready'])
        self.assertFalse(receipt['isolation_verified'])
        self.assertEqual(self.export(), (raw, receipt))

    def test_export_fails_when_input_or_selection_changes(self):
        self.dataset += b' '
        with self.assertRaises(ValueError): self.export()
        self.dataset = self.dataset[:-1]
        self.selected['generation_task_ids'][0] = self.selected['exploratory_task_ids'][0]
        with self.assertRaises(ValueError): self.export()

    def test_consumer_loads_only_bound_export_and_rejects_tampering(self):
        raw, receipt = self.export()
        ids = self.selected['generation_task_ids']
        tasks = load_generation_export(raw, expected_export_sha256=receipt['export_sha256'], task_ids=ids,
                                       expected_manifest_sha256=receipt['generation_manifest_sha256'])
        self.assertEqual(set(tasks), set(ids))
        self.assertEqual(tasks[ids[0]].id, ids[0])
        with self.assertRaises(ValueError):
            load_generation_export(raw + b' ', expected_export_sha256=receipt['export_sha256'], task_ids=ids,
                                   expected_manifest_sha256=receipt['generation_manifest_sha256'])
        with self.assertRaises(ValueError):
            load_generation_export(raw, expected_export_sha256=receipt['export_sha256'], task_ids=ids[:-1],
                                   expected_manifest_sha256=receipt['generation_manifest_sha256'])
        modified = json.loads(raw)
        modified[ids[0]]['test'][0]['output'] = [[1]]
        modified_raw = json.dumps(modified).encode()
        with self.assertRaises(ValueError):
            load_generation_export(modified_raw, expected_export_sha256=hashlib.sha256(modified_raw).hexdigest(),
                                   task_ids=ids, expected_manifest_sha256=receipt['generation_manifest_sha256'])

    def test_export_loads_into_synthetic_generation_stage(self):
        raw, receipt = self.export()
        ids = self.selected['generation_task_ids']
        tasks = load_generation_export(raw, expected_export_sha256=receipt['export_sha256'], task_ids=ids,
                                       expected_manifest_sha256=receipt['generation_manifest_sha256'])
        public, private = _quality_grids(42, 5), _quality_grids(43, 5)
        transport = Mock(return_value=FAKE_PROGRAMS[1])
        with tempfile.TemporaryDirectory() as root:
            budget = CollectionBudget.create(root, registration_sha256='a' * 64, generation=0)
            result = collect_generation(root=root, registration_sha256='a' * 64, generation=0,
                parent_adapter='none', task_ids=ids, tasks=tasks,
                expected_manifest_sha256=receipt['generation_manifest_sha256'],
                expected_task_hashes={key: task_sha256(tasks[key]) for key in sorted(ids)[:10]},
                public_probes=public, private_probes=private,
                expected_public_probes_sha256=probes_sha256(public),
                expected_private_probes_sha256=probes_sha256(private), reference_pool=[],
                expected_reference_pool_sha256=hashlib.sha256(b'[]').hexdigest(), transport=transport,
                sandbox=FakeE6Sandbox(), budget=budget)
        self.assertEqual(result['status'], 'generation_complete_unverified')
        self.assertFalse(result['training_allowed'])
        self.assertGreater(transport.call_count, 0)
