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

From the issue 2 worktree, with the repository's Python environment:

```sh
python -B - <<'PY'
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
