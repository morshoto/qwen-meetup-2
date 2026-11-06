# Independent review guide for the bounded exposure proposal

The [v3 design](design-v3.md) requires an independent review before the
candidate split is frozen. This guide is review material, not the review itself.
The proposal and checks are local; none authorizes model calls or training.

## Review inputs and scope

- [Bounded inventory proposal](bounded-exposure-proposal.json): 317 known source
  task IDs and a 200-task development candidate. It includes file hashes for the
  primary history scan, audit code, and inventory hashes for the additional
  metadata and candidate-journal scans.
- [Prompt reconciliation](prompt-reconciliation.json): three replayed checks
  covering the candidate development tasks. Read the linked individual reviews
  and their limitations; the legacy code text check had no positive match across
  the source dataset.
- [Local history search](history-search-20261105.md) and [SQLite ledger
  inventory](ledger-schema-review.json): locations searched, missing sources,
  and request ledgers whose token hashes are not attributed to source tasks.
- [Execution progress](../../docs/rsi-execution-progress.md), especially the
  five uncertain pending requests and the older agent code hash differences.

The historical inputs are partly ignored local artifacts and are not all in
Git. Review requires the corresponding retained files at the paths recorded in
the proposal. A changed or missing file requires investigation and a new
proposal; do not silently substitute another file.

## Recompute the candidate inventory

From the issue 2 worktree, with a Python environment that has the repository dependencies, including PyYAML:

```sh
python3 -B - <<'PY'
import json
from pathlib import Path
from src.arc.rsi_bounded_inventory import propose_bounded_inventory
p = json.loads(Path('research/rsi/bounded-exposure-proposal.json').read_text())
current = propose_bounded_inventory('.', task_path=p['task_source'],
    registry_path=p['prior_registry'], required_paths=p['required_paths'],
    development_size=len(p['partitions']['development']))
if current != p:
    raise SystemExit('proposal differs from current retained evidence')
print('bounded proposal matches current retained evidence')
PY
```

The reviewer must inspect the scanner coverage and inputs, separately verify the
prompt replay and ledger evidence, investigate missing or unreadable sources,
and assess whether all available task-use evidence has been accounted for.
Check the 800/200 partition, known-exposure exclusion, versioned design chain,
and that any historical records proposed for training have traceable ancestry
outside development. Record the reviewer, reviewed hashes, method, findings and
limitations in a separate review artifact. A positive review cannot establish
absence of omitted project history or model pretraining exposure; those
uncertainties must remain in the experiment report.

## Review the 750/50/200 allocation

The later [exploratory split proposal](exploratory-split-proposal.json) replaces
only the candidate allocation: 750 generation tasks, 50 exploratory tasks and
200 development tasks. The earlier 800/200 partition remains useful for
reviewing the history, but it is not the requested final allocation. The 50
exploratory tasks must remain separate from the 200, including after their
results are reviewed.

The following are the exact review inputs from the local check:

| Input | SHA-256 |
| --- | --- |
| `data/raw/arc-agi_training_challenges.json` | `779eaba89790ebad9af02514a7efc0aefaf2cf8236f046a31bbf8b9ec48f20f5` |
| `research/rsi/bounded-exposure-proposal.json` | `67b421802ffd521b18e8170b7f79b11f03ba2822879a49a17898a0fc67706512` |
| `research/rsi/exploratory-split-proposal.json` | `68ef1fc9c0c081d8f90e08506a4fcbdd795715315e01736da277314c216892de` |
| `research/rsi/audits/allocation-review.json` | `435db21430293e904f4c0074468a6bcfc953cfb4ad49beeb6bbaebc87468b024` |

Recompute the allocation in the issue worktree:

```sh
python3 -B - <<'PY'
from pathlib import Path
import hashlib
import json
from src.arc.rsi_allocation_review import review_task_allocation

paths = {
    'data/raw/arc-agi_training_challenges.json': '779eaba89790ebad9af02514a7efc0aefaf2cf8236f046a31bbf8b9ec48f20f5',
    'research/rsi/bounded-exposure-proposal.json': '67b421802ffd521b18e8170b7f79b11f03ba2822879a49a17898a0fc67706512',
    'research/rsi/exploratory-split-proposal.json': '68ef1fc9c0c081d8f90e08506a4fcbdd795715315e01736da277314c216892de',
    'research/rsi/audits/allocation-review.json': '435db21430293e904f4c0074468a6bcfc953cfb4ad49beeb6bbaebc87468b024',
}
raw = {name: Path(name).read_bytes() for name in paths}
for name, digest in paths.items():
    if hashlib.sha256(raw[name]).hexdigest() != digest:
        raise SystemExit(f'input changed: {name}')
report = json.loads(raw['research/rsi/audits/allocation-review.json'])
current = review_task_allocation(
    raw['data/raw/arc-agi_training_challenges.json'],
    raw['research/rsi/bounded-exposure-proposal.json'],
    json.loads(raw['research/rsi/exploratory-split-proposal.json']),
    expected_bounded_sha256=paths['research/rsi/bounded-exposure-proposal.json'])
if any(report.get(key) != value for key, value in current.items()):
    raise SystemExit('allocation differs from recorded review')
print('Local allocation matches the recorded report; independent review remains required.')
PY
```

Inspect the ID mapping and the canonical task content used by the check:
visible training inputs and outputs plus test inputs, without test outputs.
Verify that all 317 known-used IDs map to the corresponding source content and
are in generation, that neither evaluation set contains a known-used ID, and
that exact content hashes do not overlap among the three groups. Review the
history-scan coverage separately; matching IDs and content within the supplied
1,000 tasks does not establish that unrecorded history is absent. The remaining
433 generation tasks are only *not known* to have been used before.

Record the independent reviewer's identity, date, source commit, exact input and
code hashes, checks performed, discrepancies and missing history in a separate
review artifact. A reviewer should freeze the manifests only after resolving
any discrepancy. This guide is not a sign-off or permission to execute.
