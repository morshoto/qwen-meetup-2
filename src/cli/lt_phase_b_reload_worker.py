"""Reload one registered synthetic canary adapter under parent supervision."""
import argparse
from pathlib import Path

from src.arc.lt_batch_journal import _create
from src.arc.lt_phase_b_canary_reload import run_canary_reload
from src.arc.rsi_transport_audit import _json_object

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('canary', 'collection', 'inventory', 'allocation', 'receipt', 'summary', 'before', 'after',
                 'config', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--expected-canary-sha256', required=True)
    parser.add_argument('--expected-collection-sha256', required=True)
    args = parser.parse_args()
    result = run_canary_reload(canary_spec=_json_object(args.canary.read_bytes()),
        expected_canary_sha256=args.expected_canary_sha256,
        collection_spec=_json_object(args.collection.read_bytes()),
        expected_collection_sha256=args.expected_collection_sha256,
        source_inventory=_json_object(args.inventory.read_bytes()), source_root=ROOT,
        allocation_receipt=_json_object(args.allocation.read_bytes()), receipt=_json_object(args.receipt.read_bytes()),
        config_path=args.config, training_summary=_json_object(args.summary.read_bytes()),
        tensor_before=_json_object(args.before.read_bytes()), tensor_after=_json_object(args.after.read_bytes()))
    _create(args.output, result)


if __name__ == '__main__':
    main()
