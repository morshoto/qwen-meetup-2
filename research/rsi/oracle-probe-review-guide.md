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

The retained 14B and 32B candidates have separate [14B static metadata](audits/historical-candidates/14b-static-reference.json) and [32B static metadata](audits/historical-candidates/32b-static-reference.json). Their source and AST hashes match the retained records; verifier-only fields were excluded from the comparison. Their behavior on the proposed private probes was evaluated locally, with [14B](audits/historical-candidates/14b-behavior-reference.json) and [32B](audits/historical-candidates/32b-behavior-reference.json) fingerprint indexes. These local results are not independently audited and contain no private grids or outputs.

The historical task IDs are synthetic `e6f-...` IDs, not IDs in the proposed ARC generation partition. The existing `build_reference_pool` requires each supplied ancestor ID to belong to that generation partition, so these records cannot yet be passed to it under that contract. Do not substitute a generation task ID to satisfy the check. A separate `build_legacy_duplicate_reference` path retains the synthetic ID and has fixture coverage; it does not claim ARC ancestry or training eligibility. It was run once per model on the retained candidate and that model’s proposed private probes. Each local preparation used ten sandbox starts. Independently review its provenance, sandbox and duplicate-only policy before accepting the behavioral pool; do not automatically rerun it. This remains separate from the pending decision about whether a historical candidate may count toward the training gate.

### Recheck retained source identity

This read-only check compares each static reference with its retained E8 and E9
records. It prints no source code, private probe data, or verifier-only fields.
It is a consistency check for a reviewer; it does not replace review of the
E8/E9 seals and request provenance.

```sh
python3 -B - <<'PY'
import hashlib
import json
from pathlib import Path
from src.arc.ast_fingerprint import stable_ast_sha256

for key in ('14b', '32b'):
    index_path = Path(f'research/rsi/audits/historical-candidates/{key}.json')
    ref_path = Path(f'research/rsi/audits/historical-candidates/{key}-static-reference.json')
    index = json.loads(index_path.read_bytes())
    reference = json.loads(ref_path.read_bytes())
    if hashlib.sha256(index_path.read_bytes()).hexdigest() != reference['candidate_index_sha256']:
        raise SystemExit(f'{key}: candidate index differs')
    e9_path = Path(index['e9_run']) / 'existing_oracles.jsonl'
    e8_path = Path(index['source_campaign']) / key / 'slots/20/result.json'
    e9_raw, e8_raw = e9_path.read_bytes(), e8_path.read_bytes()
    if (hashlib.sha256(e9_raw).hexdigest() != reference['e9_existing_oracles_sha256'] or
            hashlib.sha256(e8_raw).hexdigest() != reference['e8_slot_result_sha256']):
        raise SystemExit(f'{key}: retained evidence differs')
    rows = e9_raw.splitlines()
    if len(rows) != 1:
        raise SystemExit(f'{key}: unexpected candidate count')
    row, slot = json.loads(rows[0]), json.loads(e8_raw)
    record = row['record']
    sanitized = {k: v for k, v in slot['accepted_item']['record'].items() if k != '_verifier_only'}
    source = record['messages'][-1]['content']
    if (sanitized != record or slot['verified_novel'] is not True or
            row['source_model_id'] != index['model_id'] or row['model_revision'] != index['model_revision'] or
            hashlib.sha256(source.strip().encode()).hexdigest() != reference['source_sha256'] or
            stable_ast_sha256(source) != reference['ast_sha256'] or
            reference['eligible_for_training'] is not False):
        raise SystemExit(f'{key}: reference identity differs')
    print(f'{key}: retained source identity is consistent; independent provenance review remains open')
PY
```

The earlier offline audit summaries are available in
[`e9-14b-review`](audits/e9-14b-review/summary.json),
[`e9-32b-review`](audits/e9-32b-review/summary.json), and
[`e8-rerun-review`](audits/e8-rerun-review/summary.json). Re-run or inspect the
corresponding auditors against the archived inputs in a new output location;
leave the sealed source artifacts unchanged. The local summaries do not
replace the requested independent review.

### Local preparation and budget accounting

The separate local preparation drafts are [14B](legacy-reference-preparation-14b.json) and [32B](legacy-reference-preparation-32b.json). The [preserved script](audits/legacy-reference-preparation-20261106.py) records the actual local procedure. The detailed fingerprint files remain owner-only under `artifacts/rsi-legacy-reference-preparation-20261106-v1/`; the tracked indexes contain their hashes. No model calls or training steps were used.

Each preparation used ten sandbox starts. Before any generation run, independently audit these records and ensure those starts are charged against the proposed 300-start cap if the references are used. The existing generation budget must not silently reset or extend that cap. These local preparations do not freeze the provisional probes, authorize collection, or make either historical candidate eligible for training.
