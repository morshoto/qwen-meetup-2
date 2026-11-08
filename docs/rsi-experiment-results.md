# Qwen RSI experiment: current result

The [protocol](rsi-research-protocol.md), [v2 benchmark amendment](../research/rsi/design-v2.md), [v3 exposure amendment](../research/rsi/design-v3.md), and [v4 sequencing amendment](../research/rsi/design-v4.md) describe a proposed experiment. The execution gate remains closed; the experiment has not been run. This is a report of the stopping point, not an RSI performance result.

## Evidence available

- The retained-history review identified 317 of 1,000 source tasks in known records. The provisional development split excludes those IDs. A broader metadata review found no development overlap in its recognized task identifiers, but neither review establishes complete historical prompt exposure. See the [provisional split](../research/rsi/provisional-split.json) and [extended review](../research/rsi/extended-exposure-review.json).
- An [exact-prompt review](../research/rsi/exact-prompt-exposure-review.json) found no current-format development prompts in retained JSON and JSONL string values. Other formats and missing history remain unexamined.
- A [historical prompt hash review](../research/rsi/historical-prompt-hash-review.json) found no reconstructed direct or analysis hashes for development tasks in retained JSON/JSONL values. It does not cover every historical format.
- The [legacy task-text scan](../research/rsi/historical-task-text-review.json) had no development matches and no positive match across the full source dataset, limiting its value for historical code requests.
- A [candidate journal review](../research/rsi/candidate-journal-review.json) found no development task IDs among 700 retained reservations, including interrupted candidates. Missing journals remain unexamined.
- Local fixture and synthetic checks exercise parts of collection, training preparation, monitoring and comparison. The [execution progress](rsi-execution-progress.md) records their scope and limitations. The latest full offline suite at code commit `8e4d421` reported 559 passing tests; these are not live experiment results.
- No new Qwen collection, model training, matched comparison, or independent ARC-AGI-2 evaluation was performed for this issue. Strict eligible Oracle count, model improvement, ARC-AGI-2 score and experiment compute cost are therefore not measured. No competitiveness claim can be made.

## Historical review and next experiment

- Fresh local audits of the corrected E8 rerun and E9 records support one retained historical candidate for each model, with no new strict candidates in E9. The original E8 Full run has a source dependency registration gap. See the [failure inventory](../research/rsi/audits/historical-failure-inventory.json) and separate [14B](../research/rsi/audits/historical-candidates/14b.json) and [32B](../research/rsi/audits/historical-candidates/32b.json) indexes. Whether historical candidates count in a future gate remains undecided.
- The [allocation review](../research/rsi/audits/allocation-review.json) checks the 750/50/200 split: all 317 known-used tasks fall in generation, with no exact task-content duplication among the 1,000 allocated tasks. Independent review and a freeze remain outstanding, and missing historical records limit the exposure claim. The [review guide](../research/rsi/bounded-review-guide.md) includes the current allocation inputs and a replay command.
- The [new Oracle generation proposal](../research/rsi/oracle-generation-proposal-v1.md) specifies bounded, sanitized feedback. Separate provisional probe files and a local sandbox observation are available for [independent review](../research/rsi/oracle-probe-review-guide.md). The experiment requires a separate registration and authorization; under the current draft, fewer than five eligible new same-model Oracles stops RSI training.
- Synthetic technical canary format checks were completed locally for [14B](../research/rsi/audits/technical-canary-format-14b.json) and [32B](../research/rsi/audits/technical-canary-format-32b.json), without loading weights or training. Their model file identities were also checked locally. Separate non-executing review bundles were subsequently prepared under `artifacts/rsi-canary-review-20261105-v22/14b` and `artifacts/rsi-canary-review-20261105-v22/32b`; these local ignored artifacts contain the proposed configuration and identity manifests. The earlier v1 through v21 bundles have superseded source inventories after subsequent code changes. None of these checks establishes hardware feasibility, independent verification, or permission to run. The [canary review guide](../research/rsi/technical-canary-review-guide.md) collects the current inputs and unresolved execution evidence.

A [tiny fabricated-model fixture](../research/rsi/audits/tiny-mlx-restricted-fixture-v3.json) completed one restricted MLX LoRA step and matched all 28 reloaded adapter tensors to the saved file. It used no Qwen weights or ARC data and does not establish 14B or 32B feasibility.

A [local process fixture](../research/rsi/audits/exited-group-signal-fixture.json) observed `EPERM` when signalling an already exited, unreaped owned process group. A [second fixture](../research/rsi/audits/group-membership-fixture.json) observed a zombie-only group in that case and a live member in a separate group where termination was sent. Group cleanup remains unconfirmed on this host.

## Gates before continuation

1. Independently review the bounded retained-history inventory, record missing records and uncertainty, and freeze a development split excluding all known exposures. The current reviews are local and not independently witnessed.
2. Freeze and independently audit the execution registration, including model, runtime, sandbox and transitive dependencies, prompts, data mapping, seeds and budgets. The current registration is a draft.
3. Independently register and authorize a synthetic one-step technical canary for each model, with process containment and independent adapter reload. This can occur before the Oracle gate and does not produce a promotable adapter.
4. Separately register and authorize live collection; independently verify provenance and the five strict same-model Oracle threshold before RSI adapter training.
5. If gates permit, compare against the same untrained model under matched conditions and independently score ARC-AGI-2. The 42.4% v2 target is a contextual reference for a different model and setting, not a measured result or matched-compute baseline.

If any gate cannot be satisfied, preserve this stopping point and report the specific missing evidence. The detailed [implementation status](rsi-execution-progress.md) lists additional limitations.

The [local history search](../research/rsi/history-search-20261105.md) compared
all three worktrees and reviewed earlier session records. It did not identify a
separate prompt archive or establish a complete exposure history.

The requested [50-task exploratory plan](../research/rsi/exploratory-50-plan.md) and [notebook](../nb/results/13_rsi_exploratory_50.ipynb) are prepared. No 50-task model comparison has been run; the future 14B RSI adapter is not yet available.
