"""Capture the existing conservative source tree, without authorizing execution."""
from pathlib import Path

from src.arc.lt_attestation import source_digest
from src.arc.rsi_source_inventory import build_source_inventory


def capture_source_tree(root: str | Path) -> dict:
    root = Path(root).resolve()
    before = build_source_inventory(root)
    paths = {name: root / name for name in before['files']}
    code_sha256 = source_digest(paths)
    after = build_source_inventory(root)
    if before != after:
        raise ValueError('source inventory changed during capture')
    return {'status': 'local_source_tree_unverified', 'execution_ready': False,
            'files': before['files'], 'code_sha256': code_sha256}


def verify_source_tree(root: str | Path, saved: dict) -> None:
    if capture_source_tree(root) != saved:
        raise ValueError('source tree differs from recorded inventory')


def main() -> None:
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--part', choices=('inventory', 'paths'), required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    inventory = capture_source_tree(root)
    value = inventory if args.part == 'inventory' else {name: str(root / name) for name in inventory['files']}
    print(json.dumps(value, sort_keys=True))


if __name__ == '__main__':
    main()
