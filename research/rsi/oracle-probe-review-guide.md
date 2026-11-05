# Review the proposed Oracle probes and sandbox

Status: local draft. These inputs are not independently reviewed, frozen or
authorized for collection. The four probe files are owner-only local artifacts
under `artifacts/rsi-oracle-probe-candidates-20261106-v1/`; they are not in Git.
The [probe metadata](audits/oracle-probe-candidates-20261106-v1.json) records
file and canonical content hashes without recording the private grids or their
outputs. The generation seeds were not retained; the exact files are the
proposed inputs and cannot be recreated from the report alone.

Run this check in the issue worktree without printing private probe grids:

```sh
python3 -B - <<'PY'
import hashlib
import json
import os
import stat
import subprocess
from pathlib import Path
from src.arc.e6_curriculum import canonical
from src.arc.rsi_lineage_audit import probes_sha256

report = json.loads(Path('research/rsi/audits/oracle-probe-candidates-20261106-v1.json').read_text())
source = Path('data/raw/arc-agi_training_challenges.json').read_bytes()
if hashlib.sha256(source).hexdigest() != report['task_source_sha256']:
    raise SystemExit('challenge source differs')
seen = set()
for task in json.loads(source).values():
    for pair in task['train']:
        seen.update((canonical(pair['input']), canonical(pair['output'])))
    for example in task['test']:
        seen.add(canonical(example['input']))
for model in ('14b', '32b'):
    for kind in ('public', 'private'):
        metadata = report['probes'][model][kind]
        path = Path(metadata['local_path'])
        if path.is_symlink() or not path.is_file():
            raise SystemExit(f'{model} {kind}: file unavailable')
        info = path.stat()
        if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600:
            raise SystemExit(f'{model} {kind}: file permissions differ')
        raw = path.read_bytes()
        grids = json.loads(raw)
        if (hashlib.sha256(raw).hexdigest() != metadata['file_sha256'] or
                probes_sha256(grids) != metadata['canonical_probe_sha256'] or len(grids) != 5):
            raise SystemExit(f'{model} {kind}: content differs')
        keys = {canonical(grid) for grid in grids}
        if len(keys) != 5 or keys & seen:
            raise SystemExit(f'{model} {kind}: exact grid overlap')
        seen.update(keys)
    print(f'{model}: local probe hashes and exact nonoverlap match')
sandbox = json.loads(Path('research/rsi/audits/oracle-sandbox-observation-20261106-v1.json').read_text())
image_id = subprocess.check_output(['docker', 'image', 'inspect', sandbox['image_tag'],
                                    '--format', '{{.Id}}'], text=True, timeout=10).strip()
if image_id != sandbox['image_id']:
    raise SystemExit('sandbox image differs')
print('Local sandbox image ID matches; containment is not established.')
PY
```

Review the sampled grid method, exact source and selection hashes, separate
model inventories, and the image and container execution limits. Check the
[local image observation](audits/oracle-sandbox-observation-20261106-v1.json),
including its runner readback and limitations. Exact nonoverlap covers the
available challenge inputs and visible training outputs; it does not establish
semantic independence, complete history, or separation from unseen data.
Private probes and their outputs must not be sent to Qwen. The duplicate
reference pool still needs construction and independent provenance review;
the historical candidate's training-gate policy remains a separate decision.
Record the reviewer, input and code hashes, findings, missing artifacts, and
remaining uncertainty before freezing a new execution registration.

## Historical duplicate references

The retained 14B and 32B candidates have separate [14B static metadata](audits/historical-candidates/14b-static-reference.json) and [32B static metadata](audits/historical-candidates/32b-static-reference.json). Their source and AST hashes match the retained records; verifier-only fields were excluded from the comparison. Their behavior on the newly proposed private probes has not been evaluated.

The historical task IDs are synthetic `e6f-...` IDs, not IDs in the proposed ARC generation partition. The existing `build_reference_pool` requires each supplied ancestor ID to belong to that generation partition, so these records cannot yet be passed to it under that contract. Do not substitute a generation task ID to satisfy the check. A separate `build_legacy_duplicate_reference` path retains the synthetic ID and has fixture coverage; it does not claim ARC ancestry or training eligibility. It has not been run on the retained candidates or new probes. Independently review its provenance, sandbox and duplicate-only policy before constructing a behavioral pool. This remains separate from the pending decision about whether a historical candidate may count toward the training gate.
