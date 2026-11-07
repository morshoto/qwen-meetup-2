"""Run one registered public comparison task under parent supervision."""
import argparse
from pathlib import Path

from src.arc.lt_batch_journal import _create
from src.arc.lt_phase_b_registered_comparison import run_registered_comparison_task
from src.arc.rsi_transport_audit import _json_object

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('comparison', 'collection', 'training', 'canary', 'inventory', 'allocation', 'collection-receipt',
                 'comparison-receipt', 'export', 'run-dir', 'model-path', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('comparison', 'collection', 'training', 'canary'):
        parser.add_argument('--expected-' + name + '-sha256', required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--adapter-state', required=True, choices=('off', 'on'))
    args = parser.parse_args()
    result = run_registered_comparison_task(comparison_spec=_json_object(args.comparison.read_bytes()),
        expected_comparison_sha256=args.expected_comparison_sha256, collection_spec=_json_object(args.collection.read_bytes()),
        expected_collection_sha256=args.expected_collection_sha256, training_spec=_json_object(args.training.read_bytes()),
        expected_training_sha256=args.expected_training_sha256, canary_spec=_json_object(args.canary.read_bytes()),
        expected_canary_sha256=args.expected_canary_sha256, source_inventory=_json_object(args.inventory.read_bytes()),
        source_root=ROOT, run_dir=args.run_dir, model_path=args.model_path,
        allocation_receipt=_json_object(args.allocation.read_bytes()), collection_receipt=_json_object(args.collection_receipt.read_bytes()),
        comparison_receipt=_json_object(args.comparison_receipt.read_bytes()), export=_json_object(args.export.read_bytes()),
        task_id=args.task, adapter_state=args.adapter_state)
    _create(args.output, result)


if __name__ == '__main__':
    main()
