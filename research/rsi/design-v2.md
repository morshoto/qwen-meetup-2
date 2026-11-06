# Qwen RSI research design v2: benchmark amendment

This amendment supersedes the **26.0% competitive reference** in the
[version 1 design](../../docs/rsi-research-protocol.md). All other v1 design
constraints and unresolved gates continue to apply. The design remains a local,
reviewable proposal: it is not an executable preregistration, an independently
witnessed timestamp, an approval for model calls or training, or evidence that
RSI works. The earlier [v1 registration](design-registration-v1.json) remains
preserved as history.

## Updated reference and target

The preregistered research target for the planned candidate is **at least 42.4%**
on independently verified ARC-AGI-2 semi-private evaluation. This is the score
reported by [ARC Prize for Qwen3.8-27B at xhigh reasoning effort](https://arcprize.org/results/alibaba-qwen3-8-27b).
The [Qwen model repository](https://huggingface.co/Qwen/Qwen3.8-27B) provides
open weights. ARC Prize reports an estimated **$0.4469 per task**, calculated
from recorded token usage and September 30, 2026 Alibaba Cloud rates. That is a
pricing estimate, not the actual charge for its Baseten evaluation.

The reference model is a later, larger model, run with a different reasoning
setting, inference provider, model format and budget. Its score is contextual
benchmark evidence, not a matched-compute comparison or a Qwen3-14B baseline.
Meeting 42.4% does not establish equal efficiency. The planned matched comparison
against the untrained Qwen3-14B base and each generation's immediate parent is
still required to support improvement claims. Report local hardware, inference
cost, total training cost and resource use separately; don't equate local
resource estimates with the reference's token-price estimate.

The v1 reference, Efficient Evolutionary Program Synthesis at 26.0% on the
semi-private set, remains a valid historical data point with a different method
and budget. It is not the selected competitiveness threshold for v2. ARC Prize
also reports the open-source NVARC competition system at 24.03% on the 2025
**private** evaluation set; that is useful historical context, not an
interchangeable result on the semi-private set. See the [2025 results](https://arcprize.org/blog/arc-prize-2025-results-analysis).

## Before execution

The v1 split is provisional and its retained-history inventory is not proven
complete. Freeze and audit the final split, task-conditioned implementation,
model and verifier dependencies, exact prompts and seeds, and resource budgets
before any new live experiment. Obtain the separately scoped authorizations
listed in v1. If independent semi-private scoring is unavailable, do not claim
the target was met. Any change to this target or method requires a new version
recorded before the affected evaluation; do not revise it after observing scores.
