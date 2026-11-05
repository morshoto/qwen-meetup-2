# Qwen RSI research design v3: bounded exposure review

This amendment supersedes only the complete-history stop condition in the
[version 1 protocol](../../docs/rsi-research-protocol.md), retaining the
[version 2 benchmark target](design-v2.md) and all other requirements. The
user accepted the [proposal](exposure-gate-amendment-proposal.md). This is a
prospective design record, not an execution registration, independent review,
authorization for model calls or training, or evidence that RSI works.

## Exposure and split

Before affected execution, independently review a bounded inventory of all
available historical task-use records. Freeze the inventory and audit code
hashes, record searched locations and missing or unreadable sources, and exclude
all known exposed tasks. Select the last 200 sorted eligible source IDs for
development and assign the rest to generation, with separate manifest hashes.
Stop if fewer than 200 eligible IDs remain, a required available record cannot
be reviewed, or the review and manifests have not been frozen. Exclude prior
generation records with missing ancestry or development-overlapping ancestry
from training.

Disclose **unknown prior project exposure** separately from unknown pretraining
exposure. If newly discovered historical records overlap the frozen development
manifest before evaluation, stop and issue a new prospective registration. Do
not revise the split after seeing comparison outcomes to improve a result.

## Interpretation and remaining gates

A matched comparison can describe a paired difference under the frozen
conditions; it cannot establish that tasks were unseen in all previous project
activity or pretraining. The current split remains provisional pending the
bounded independent review and manifest freeze. Execution requires the other
registered provenance, dependency, budget, process, preflight, selection and
separate authorization gates. The 42.4% independent ARC-AGI-2 reference stays
unchanged and remains contextual rather than matched compute. If independent
scoring is unavailable, competitiveness remains unverified.
