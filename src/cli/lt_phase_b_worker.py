"""Run one public registered Phase B task; parent supervision is required."""
import argparse
from pathlib import Path

from src.arc.lt_batch_journal import _create
from src.arc.lt_phase_b_registered_worker import run_registered_task
from src.arc.rsi_transport_audit import _json_object

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('spec', 'inventory', 'export', 'receipt', 'allocation', 'output', 'model-path'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--expected-run-sha256', required=True)
    args = parser.parse_args()
    result = run_registered_task(spec=_json_object(args.spec.read_bytes()),
        expected_run_sha256=args.expected_run_sha256, source_inventory=_json_object(args.inventory.read_bytes()),
        source_root=ROOT, model_path=args.model_path, export=_json_object(args.export.read_bytes()),
        receipt=_json_object(args.receipt.read_bytes()), allocation_receipt=_json_object(args.allocation.read_bytes()),
        task_id=args.task)
    _create(args.output, result)


if __name__ == '__main__':
    main()
