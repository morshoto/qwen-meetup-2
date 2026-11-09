import copy
import json

import pytest

from src.arc.qwen38_weight_verification_receipt import build_receipt, verify_receipt, write_receipt
from test_qwen38_weight_fresh_verify import fixture


def test_receipt_binds_journal_and_returns_only_outcomes(tmp_path):
    manifest, challenges, solutions, rows = fixture()
    receipt = build_receipt(manifest, challenges, solutions, rows)
    assert receipt['outcomes'] == [{'task_id': 'a', 'candidate_index': 0, 'correct': True}]
    assert '[[[7]]]' not in json.dumps(receipt)
    path = tmp_path / 'receipt.json'
    write_receipt(path, receipt)
    assert verify_receipt(json.loads(path.read_text()), manifest, challenges, solutions, rows) == {('a', 0): True}
    with pytest.raises(FileExistsError): write_receipt(path, receipt)


def test_receipt_replay_rejects_changed_results_journal_and_sources():
    manifest, challenges, solutions, rows = fixture()
    receipt = build_receipt(manifest, challenges, solutions, rows)
    altered = copy.deepcopy(receipt); altered['outcomes'][0]['correct'] = False
    with pytest.raises(ValueError): verify_receipt(altered, manifest, challenges, solutions, rows)
    changed = copy.deepcopy(rows); changed[0]['raw'] = '[[[0]]]'
    with pytest.raises(ValueError): verify_receipt(receipt, manifest, challenges, solutions, changed)
    with pytest.raises(ValueError): verify_receipt(receipt, manifest, challenges, solutions + b' ', rows)
