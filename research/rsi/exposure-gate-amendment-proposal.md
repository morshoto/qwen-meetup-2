# Proposal: proceed with documented exposure uncertainty

**Status: draft for review.** This proposal does not amend the registered
protocol, freeze a split, authorize a model call or approve training. The
existing [stop condition](../../docs/rsi-research-protocol.md) remains in force.

## Proposed change

Replace the requirement to prove a complete historical exposure inventory with
a bounded, independently reviewed inventory of the records that can be found.
Before any affected execution, freeze the inventory and audit code hashes,
record the locations searched and missing or unreadable sources, and exclude
all known exposed tasks. Freeze a development manifest only if at least 200
source tasks remain. Exclude historical generation records with missing or
development-overlapping ancestry from training.

Record the uncertainty as **unknown prior project exposure**, separately from
unknown model pretraining exposure. Reconstruct and independently review any
newly discovered historical records; if they overlap a frozen development
manifest before evaluation, stop and issue a new prospective registration. Do
not revise the split after seeing comparison outcomes to improve the result.

## Interpretation

A matched trained-versus-base comparison could describe a paired difference on
the frozen tasks under the recorded conditions. It would not establish that
those tasks were unseen in all historical project activity or model pretraining.
The independently scored ARC-AGI-2 target and all other provenance, budget,
safety, preflight, selection and separately scoped authorization requirements
remain in force. If independent scoring is unavailable, competitiveness remains
unverified.

## Review decision needed

Accepting this proposal changes the issue 1 research design and its evidence
standard. A new version must be recorded and audited before the affected
experiment; the current v1 and v2 records stay intact. If the change is not
accepted, the existing incomplete-history stop condition continues to apply.
