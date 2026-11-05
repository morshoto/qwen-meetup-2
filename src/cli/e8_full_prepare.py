"""Read-only preflight and explicit pre-registration for E8 Full."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.arc.e8_config import ORDER, load_e8_config
from src.arc.e8_resource import check_resources, snapshot
from src.pipelines.e5 import _docker_image_id, _seal_mutable_file, _verify_mutable_file
from src.pipelines.e6 import _json
from src.pipelines.e6f import _sha
from src.pipelines.e6g import PROJECT, _verify_g
from src.pipelines.e8 import _base, _check_no_other_model_server, _protocol, _source_pool


def prepare(config: Path, smoke: Path, campaign: Path | None = None, exclude: Path | None = None) -> dict:
    settings = load_e8_config(config)
    if settings.profile != 'full' or settings.slots != 10:
        raise ValueError('requires a ten-slot Full profile')
    smoke = smoke.resolve()
    lock = smoke / 'protocol.lock.yml'
    _verify_mutable_file(lock)
    if (smoke / 'protocol.sha256').read_text().strip() != _sha(lock):
        raise ValueError('Smoke protocol hash mismatch')
    previous = json.loads(lock.read_text())
    used = {slot['slot'] for slot in previous['slots']}
    used_seeds = {seed for slot in previous['slots'] for seed in slot['attempt_seeds']}
    excluded_hash = None
    if exclude is not None:
        excluded = exclude.resolve() / 'protocol.lock.yml'
        _verify_mutable_file(excluded)
        if (excluded.parent / 'protocol.sha256').read_text().strip() != _sha(excluded):
            raise ValueError('excluded protocol hash mismatch')
        prior = json.loads(excluded.read_text())
        used.update(slot['slot'] for slot in prior['slots'])
        used_seeds.update(seed for slot in prior['slots'] for seed in slot['attempt_seeds'])
        excluded_hash = _sha(excluded)
    for stage in ORDER:
        _verify_g(smoke / stage)
        if json.loads((smoke / stage / 'result.json').read_text())['status'] != 'completed':
            raise ValueError('Smoke stage incomplete')
    _check_no_other_model_server()
    resource = snapshot(PROJECT)
    reason = check_resources(resource, resource, min_free_percent=settings.min_memory_free_percent,
        max_swap_increase_mib=settings.max_swap_increase_mib,
        min_disk_bytes=settings.min_free_disk_gib * 1024**3)
    if reason: raise RuntimeError(reason)
    base = _base(settings)
    image = _docker_image_id(base)
    pool, _, _, splits, identity = _source_pool(settings, base, image)
    protocol = _protocol(settings, base, image, identity)
    if used & {s['slot'] for s in protocol['slots']} or used_seeds & {
            seed for slot in protocol['slots'] for seed in slot['attempt_seeds']}:
        raise ValueError('Full slots or seeds overlap Smoke')
    if 'src/arc/training.py' not in identity['implementation_sha256']:
        raise ValueError('training dependency absent')
    for stage in ORDER:
        recorded = json.loads((smoke / stage / 'result.json').read_text())['model_manifest']
        if protocol['model_file_manifests'][stage] != {k: v for k, v in recorded.items() if k != 'model_id'}:
            raise ValueError('model files differ from Smoke: ' + stage)
    summary = {'smoke_protocol_sha256': _sha(lock), 'slots': protocol['slots'],
               'excluded_protocol_sha256': excluded_hash,
               'resource': resource, 'sandbox_image_id': image,
               'model_revisions_and_hashes_verified': True,
               'legacy_oracles_strict_eligible': 0, 'training_enabled': False}
    if campaign is not None:
        campaign = campaign.resolve()
        if not campaign.is_relative_to((PROJECT / 'artifacts/e8').resolve()):
            raise ValueError('campaign must be inside artifacts/e8')
        campaign.mkdir(mode=0o700, parents=True, exist_ok=False)
        for name, value in (('protocol.lock.yml', protocol), ('initial_pool.json', pool),
                            ('split_manifest.json', splits), ('preflight.json', summary)):
            path = campaign / name
            _json(path, value, private=True)
            _seal_mutable_file(path)
        (campaign / 'protocol.sha256').write_text(_sha(campaign / 'protocol.lock.yml') + '\n')
        (campaign / 'protocol.sha256').chmod(0o600)
        summary['protocol_sha256'] = _sha(campaign / 'protocol.lock.yml')
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--smoke', type=Path, required=True)
    parser.add_argument('--campaign-dir', type=Path, help='new directory; omit for read-only check')
    parser.add_argument('--exclude-campaign', type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.config, args.smoke, args.campaign_dir, args.exclude_campaign), indent=2))


if __name__ == '__main__': main()
