"""Reload a registered real adapter under parent process supervision."""
import argparse
from pathlib import Path

from src.arc.lt_batch_journal import _create
from src.arc.lt_phase_b_real_reload import run_real_reload
from src.arc.rsi_transport_audit import _json_object

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('training', 'canary', 'collection', 'inventory', 'allocation', 'receipt', 'run-dir', 'config', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('training', 'canary', 'collection'):
        parser.add_argument('--expected-' + name + '-sha256', required=True)
    args = parser.parse_args()
    result = run_real_reload(training_spec=_json_object(args.training.read_bytes()),
        expected_training_sha256=args.expected_training_sha256, canary_spec=_json_object(args.canary.read_bytes()),
        expected_canary_sha256=args.expected_canary_sha256, collection_spec=_json_object(args.collection.read_bytes()),
        expected_collection_sha256=args.expected_collection_sha256, source_inventory=_json_object(args.inventory.read_bytes()),
        source_root=ROOT, run_dir=args.run_dir, allocation_receipt=_json_object(args.allocation.read_bytes()),
        receipt=_json_object(args.receipt.read_bytes()), config_path=args.config)
    _create(args.output, result)


if __name__ == '__main__':
    main()
