# Independent technical canary review

Status: review packet, not approval or authorization. The proposed 14B and
32B canaries are separate one-step synthetic-only runs. No adapter from either
run exists, and no hardware feasibility has been established. Review each
model separately; a successful 14B run does not approve or prove a 32B run.

## Current draft inputs

- [14B registration](technical-canary-registration-14b-v17.json)
- [32B registration](technical-canary-registration-32b-v17.json)
- [Reload review procedure](technical-canary-reload-review.md)
- [Tiny fabricated-model fixture](audits/tiny-mlx-restricted-fixture-v2.json)
- [Exited group signal fixture](audits/exited-group-signal-fixture.json) and
  [group membership fixture](audits/group-membership-fixture.json)

The model, tokenizer, runtime, environment, source and data identities in these
drafts were checked locally. The bundles live under ignored `artifacts/` paths
on this machine; they are not included in Git and can drift. Their existence
and hashes do not establish independent review or process provenance.

## Recheck the draft files

In the issue worktree, run the following without loading model weights:

```sh
python3 -B - <<'PY'
import hashlib
import json
from pathlib import Path

for key in ('14b', '32b'):
    path = Path(f'research/rsi/technical-canary-registration-{key}-v17.json')
    registration = json.loads(path.read_bytes())
    root = Path(registration['local_root'])
    for name, expected in registration['local_file_sha256'].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise SystemExit(f'{key}: file changed: {name}')
    if (root / 'nonpromotable.adapter').exists():
        raise SystemExit(f'{key}: proposed adapter path already exists')
    if (registration['execution_ready'] is not False or
            registration['training_allowed'] is not False or
            registration['promotion_allowed'] is not False or
            registration['independently_verified'] is not False):
        raise SystemExit(f'{key}: draft status differs')
    print(f'{key}: listed local files match; no adapter exists')
PY
```

A reviewer must independently check the pinned model and source manifests,
runtime, synthetic dataset and its overlap checks, exact one-step config,
separate output paths, offline settings, and training and reload resource
limits. Recheck identities immediately before any proposed execution.

## Unresolved before a run can be claimed as verified

- The local `sandbox-exec` fixture prevents ordinary fork and spawn, and a tiny
  random model completed one restricted step and matched its reloaded factors. The tiny reload subprocess had a timeout but no resource sampling. This does not prove containment
  of escaped or externally created work, or establish Qwen model compatibility.
- On this host, group signals returned `EPERM` for an exited unreaped leader.
  Group listings provide observations but do not prove cleanup. Agree and
  independently review how the owned process and any descendants are accounted
  for before interpreting a canary as successful.
- The sampler and loss monitor have fixture coverage; their observations need
  independently verifiable process provenance. Register and review the
  separate reload budget and observation procedure.
- Record the reviewer, date, source commit, manifest hashes, checks, findings,
  unresolved limitations and decision in a separate artifact. A separate,
  scoped authorization is required for a real model canary. No review or
  authorization for a synthetic canary authorizes RSI training.

If a canary is authorized and run, preserve the evidence and adapter in
quarantine. A one-step adapter cannot be promoted or used as a research
candidate. A failure or inconclusive observation is a result, not permission
to extend the resource limits or rerun automatically.
