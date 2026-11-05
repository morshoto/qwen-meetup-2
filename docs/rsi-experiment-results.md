# Qwen RSI experiment: current result

The [protocol](rsi-research-protocol.md) and [v2 benchmark amendment](../research/rsi/design-v2.md) and [v3 exposure amendment](../research/rsi/design-v3.md) describe a proposed experiment. The execution gate remains closed; the experiment has not been run. This is a report of the stopping point, not an RSI performance result.

## Evidence available

- The retained-history review identified 317 of 1,000 source tasks in known records. The provisional development split excludes those IDs. A broader metadata review found no development overlap in its recognized task identifiers, but neither review establishes complete historical prompt exposure. See the [provisional split](../research/rsi/provisional-split.json) and [extended review](../research/rsi/extended-exposure-review.json).
- An [exact-prompt review](../research/rsi/exact-prompt-exposure-review.json) found no current-format development prompts in retained JSON and JSONL string values. Other formats and missing history remain unexamined.
- A [historical prompt hash review](../research/rsi/historical-prompt-hash-review.json) found no reconstructed direct or analysis hashes for development tasks in retained JSON/JSONL values. It does not cover every historical format.
- The [legacy task-text scan](../research/rsi/historical-task-text-review.json) had no development matches and no positive match across the full source dataset, limiting its value for historical code requests.
- A [candidate journal review](../research/rsi/candidate-journal-review.json) found no development task IDs among 700 retained reservations, including interrupted candidates. Missing journals remain unexamined.
- Local fixture and synthetic checks exercise parts of collection, training preparation, monitoring and comparison. The [execution progress](rsi-execution-progress.md) records their scope and limitations. The latest full offline suite at code commit `1e74689` reported 538 passing tests; these are not live experiment results.
- No new Qwen collection, model training, matched comparison, or independent ARC-AGI-2 evaluation was performed for this issue. Strict eligible Oracle count, model improvement, ARC-AGI-2 score and experiment compute cost are therefore not measured. No competitiveness claim can be made.

## Gates before continuation

1. Independently review the bounded retained-history inventory, record missing records and uncertainty, and freeze a development split excluding all known exposures. The current reviews are local and not independently witnessed.
2. Freeze and independently audit the execution registration, including model, runtime, sandbox and transitive dependencies, prompts, data mapping, seeds and budgets. The current registration is a draft.
3. Obtain separately scoped authorization before live collection; independently verify provenance and the five strict same-model Oracle threshold before training.
4. Establish and authorize the one-step preflight, including process containment and independent adapter reload; audit its observations before any training stage.
5. If gates permit, compare against the same untrained model under matched conditions and independently score ARC-AGI-2. The 42.4% v2 target is a contextual reference for a different model and setting, not a measured result or matched-compute baseline.

If any gate cannot be satisfied, preserve this stopping point and report the specific missing evidence. The detailed [implementation status](rsi-execution-progress.md) lists additional limitations.

The [local history search](../research/rsi/history-search-20261105.md) compared
all three worktrees and reviewed earlier session records. It did not identify a
separate prompt archive or establish a complete exposure history.

The requested [50-task exploratory plan](../research/rsi/exploratory-50-plan.md) and [notebook](../nb/13_rsi_exploratory_50.ipynb) are prepared. No 50-task model comparison has been run; the future 14B RSI adapter is not yet available.
