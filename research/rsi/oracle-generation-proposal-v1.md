# Prospective Oracle-generation pilot

**Status: design proposal, not an execution registration or authorization.**
This task-conditioned method is a new experiment distinct from E9. Its purpose
is to measure whether the changed prompts and bounded public feedback can
produce useful, novel candidates; it does not presume that ten slots will
produce five eligible Oracles.

## Evidence motivating the pilot

In the corrected E8 rerun, 14B had four static-safety rejections among ten
slots; 32B had six private-acceptance rejections. In the independently audited
E9 runs, 14B's 30 slots had 15 static-safety rejections and 15 valid but
non-novel candidates; 32B's ten slots had four private-acceptance rejections
and six valid but non-novel candidates. E9 produced zero new strict Oracles in
both runs. The [failure inventory](audits/historical-failure-inventory.json)
retains categories and source hashes. Raw/AST duplicate flags overlap and
finite behavioral fingerprints do not prove semantic equivalence. These small
historical samples do not establish that this pilot will reach the gate.

## Method and feedback

For each model, independently select the first ten registered generation tasks
for generation 0. Show the training examples and test inputs, never test
outputs, and request one complete `transform(grid)` program. Use the existing
task-conditioned prompt and fixed Thinking-off settings, then allow at most
two model-authored repairs. No human-authored ARC transformation, operator
search or task-specific solving heuristic is supplied.

A repair may use only the prior candidate source and an allowed generic
category: `syntax_error`, `parse_or_contract_error`, `runtime_exception`,
`shape_or_color_violation`, `visible_demonstration_mismatch`,
`duplicate_source` or `duplicate_ast`. Quote the previous code as untrusted
data and give no private probe input, output, correct answer, or private
behavior-match reason. A prohibited operation, private-acceptance failure,
private behavioral match, nondeterminism, repeated source or infrastructure
failure does not receive repair feedback; handle it as a rejection or stop
according to its registered category. The prompt and exact category policy must
be frozen and audited before use.

Screen candidates with the registered safety checks, sandbox, public and
private probes and duplicate reference pool. Keep source/AST matches distinct
from sampled behavioral matches. Count every attempt, valid candidate,
duplicate, invalid result and strict-new candidate; retain local request,
response, sandbox, resource and rejection evidence separately for each model.
Program-generated outputs are synthetic generator distillation, not verified
real ARC answers.

## Proposed limits and stop rules

Per model, per generation: ten task slots, at most three attempts per slot,
32 model calls including two setup canaries, 200,000 reserved tokens, 300
sandbox starts and 3,600 seconds. Use the existing schedule and seeds only
once a new model-specific execution registration has frozen them. No unused
budget transfers to the other model or stage. These limits are ceilings, not
proof that five candidates can be collected within them.

The 14B and 32B runs require separate model identity, dependency, budget and
provenance review and separate execution authorization. The current collector
and identity integrations must be audited for each model; 32B support is not
implied by the 14B path. Stop on a budget, infrastructure, provenance or
resource failure. A model with fewer than five independently audited **new**
same-model Oracles stops before RSI training. Do not automatically extend the
run or count an older Oracle; a change requires a new prospective registration.
The proposed 750/50/200 manifests remain provisional pending independent
review and freeze.
