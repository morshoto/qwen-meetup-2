# Issue 2: TDD execution plan

The design in [issue 1](rsi-research-protocol.md) and its
[v2 amendment](../research/rsi/design-v2.md) are design records, not execution
authorization. This issue implements the method and audit, then records either
an authorized audited outcome or the gate that prevented continuation.

## Test-first cycles

1. **Task-conditioned generation:** first test that prompts contain only the
   selected training demonstrations and test inputs, with no test labels;
   test deterministic task selection, seed scheduling, and sanitized repair
   feedback. Implement the smallest prompt and scheduling interfaces.
2. **Candidate and data verification:** first test sandbox-only execution,
   visible-example checking, generated-label provenance, private-probe isolation,
   duplicate rejection, and the five-distinct-record gate. Implement the bounded
   collector and fail-closed data audit using the existing verifier contracts.
3. **Registration and lifecycle:** first test dependency-closed hashes, immutable
   inputs, resource limits, authorization boundaries, and interrupted/resume
   behavior. Implement registration, guarded collection, and independent audit;
   missing evidence must prevent live dispatch.
4. **Training and comparison:** first test monitoring and no-training skip,
   parent adapter identity, matched candidate/base evaluation, scoring and paired
   uncertainty. Implement preflight, training, comparison, and artifact reporting
   under separately authorized budgets. Test fixture behavior separately from
   actual model and score evidence.
5. **Offline integration:** run fake end-to-end tests through each gate, the
   focused suites, and the full available suite. Review the exact head and
   record results and any gate preventing authorized execution.

No cycle authorizes model calls, training, cloud resources, or benchmark
submission. Do not change sealed prior artifacts, reuse uncertain results as new,
expose hidden labels in prompts, or use human-authored ARC solving heuristics.
Live resource feasibility and independent semi-private scoring remain unproven.

### Shared collection budget cycle

Test a per-generation durable budget with atomic concurrent reservations for
all model calls including canaries, reserved tokens, sandbox starts and elapsed
wall time. Test exhaustion, restart, clock rollback, and fail-closed persistence.
Implement it as a cooperating local guard; independent authentication and a
final registered budget remain separate prerequisites for live execution.

### Conditioned slot orchestration cycle

Test a scheduled slot with a supplied transport and sandbox: matching frozen
input hashes, pre-dispatch call and token reservations, bounded public repairs,
terminal private failures, novelty screening, and durable completion evidence.
Test interrupted attempts remain consumed. Treat supplied components and local
records as unauthenticated, and keep drafts ineligible for training.

### Captured transport binding cycle

Test the conditioned collector against a local fixture HTTP server, checking
wire seeds, thinking mode, token reservation, deadlines, captures, and failures
before dispatch for invalid settings. Bind the collector to the capturing backend
without enabling or claiming an authenticated live run.

### Capture-to-slot binding cycle

Test that each completed captured attempt records its unique transport request
ID in the slot evidence, and that malformed or repeated IDs fail closed. Use
these identifiers for later replay across interleaved logs; they are not proof
of server identity or an independent seal.

### Captured slot replay cycle

Test hash-bound slot evidence against captured request, response and ledger
records joined by request ID, including reordered logs, missing or duplicate
IDs, changed bytes, incorrect statuses, and novelty fingerprints. Replay the
conditioned lineage with a separately supplied sandbox; keep the result
ineligible until server and sandbox provenance are independently established.

### Matched evaluation task cycle

Test that two independent samples use the same label-free solve prompt, that
static rejection, visible-example failures and invalid grid outputs count as
failed attempts, and that infrastructure failures and the ten-start per-sample
cap stop scoring. Test per-test-input guesses and task scoring via the isolated
solution verifier. Keep supplied model and sandbox identity unauthenticated.

### Frozen paired comparison cycle

Test that exactly 200 matched task outcomes and registered per-task seeds are
frozen before bootstrapping, that parent and candidate adapter identities are
bound, and that incomplete, unmatched or altered records cannot yield a paired
summary. Treat local records and claimed execution settings as unauthenticated.

### Comparison budget and execution cycle

Test durable per-comparison ceilings of 800 model calls, 16 million reserved
tokens, 8,000 sandbox starts, and 36,000 seconds; exclude setup canaries from
this budget. Then test a matched two-arm runner with a frozen task and seed
schedule, shared budget, incomplete-run reporting, and output freeze. Fixtures
must not be interpreted as authenticated model results.

### Comparison replay cycle

Test that task records retain bounded sample source and can be replayed against
separately supplied tasks, sandbox and verifier. Replay all 400 task-arm records
and check their scores against the frozen outcomes. Reject missing or changed
records; the local replay does not prove model, runtime, adapter or run origin.

### Training preparation cycle

Test that a prospective batch is bound to canonical task prompts, distinct
quarantined sources, the parent adapter, and fixed QLoRA settings. A separate
one-step preflight plan must not consume the generation batch or authorize any
training. Missing independent audit, execution registration, process-containment
proof and explicit authorization must remain visible blockers.

### RSI SFT format and mask cycle

Test conversion of quarantined records to the MLX chat-row shape and verify
that the thinking-off inference prefix exactly matches the masked token prefix,
leaving nonempty completion tokens within the sequence limit. Reject altered
batch hashes, task prompts, source bindings and token offsets. The supplied
tokenizer and renderer remain unauthenticated until separately checked against
the registered runtime and model.

### Local tokenizer runtime audit cycle

Test a read-only tokenizer audit that binds the proposed snapshot path,
tokenizer files and prepared batch to explicit hashes, checks the MLX-LM
version, and runs the format precheck with the actual ChatDataset processor.
Keep the audit result separate from model inference, training authorization and
independent batch provenance. Exercise the local tokenizer with synthetic
fixture records if the pinned runtime is available.

### Comparison capture binding cycle

Test local guarded HTTP transport for both comparison arms with matched seeds,
settings and per-arm capture logs. Bind each returned request ID to the saved
task sample; reject mismatched adapter routing or repeated IDs. The endpoint's
asserted adapter remains unauthenticated until independently checked.

### Comparison capture replay cycle

Test all 800 sample records against their per-arm request, response and ledger
captures, joined by unique request ID. Replay the saved programs and scores,
reject mismatched seeds, prompts, source bytes, missing or duplicate records,
and preserve the distinction between capture consistency and authenticated
server, adapter, sandbox or model provenance.

### Comparison server contract cycle

Test both local guarded endpoints before collection and again before dispatch,
checking the declared model, adapter, thinking canary, reviewed driver, distinct
ledgers and combined token ceilings. Reject changing or regressing contract
state. A server declaration is not process or weight attestation.

### Local model and adapter identity cycle

Test complete local snapshot file hashing, quantization and weight-index checks,
chat-template binding, file drift and escaping symlink rejection. Test the
existing two-file adapter digest against registered expectations. These local
hashes are evidence of files read, not proof of server process identity or an
independently witnessed registration.

### Registered model consistency cycle

Test that the draft registration's model-manifest bytes match the actual local
snapshot, and detect changes to either the registered inputs or model files.
Recheck registration before and after the file audit. This is not a final
preregistration, remote witness, or server process attestation.

### Captured collection to draft cycle

Test that a captured slot must pass replay before it can produce a quarantined
SFT draft, and that the draft binds the final captured request, response hash,
source, task and novelty result. Mark origin as unverified; the existing batch
gate must continue to reject the draft until independent provenance is proven.

### Invalid evaluation response cycle

Test that a captured truncated or malformed model completion consumes one
sample and counts as an invalid attempt, while malformed transport, budget and
identity failures make the comparison incomplete. Bind invalid outcomes to
captured response bytes during replay rather than treating them as valid code.

### Invalid collection response cycle

Test that truncated or malformed collection completions consume a single slot,
remain ineligible, and cannot become repair input or a draft. Replay the captured
bytes to verify the invalid outcome; infrastructure failures remain uncertain
and stop collection.

### Collection generation cycle

Test ten scheduled slots with a single stage budget and a reference pool that
incorporates earlier provisionally novel slots. Bind the task, probe, parent and
reference identities before starting. Stop on infrastructure failure, preserve
completed slot outcomes, and never authorize training from local results.

### Generation capture replay cycle

Test stage-wide replay of all ten slot journals and captured request records,
reconstructing the novelty pool in order and checking the stage counts and
completion markers. Reject altered, missing or duplicate slot evidence. The
replay remains local and does not authenticate server, model or sandbox origin.

### Generation draft batch cycle

Test stage replay followed by slot-ordered draft derivation using the evolving
reference pool. Record the provisional draft count and the five-record
threshold without marking any draft independently audited or training eligible.
Reject altered captures and preserve private probe and query-output boundaries.

### Local runtime identity cycle

Test a read-only native interpreter probe that checks the proposed MLX-LM
version, reviewed generation source, Python executable digest and selected
package versions. Bind the observation to the draft runtime manifest and detect
drift. A local probe does not prove complete dependency closure, process
identity, live execution authorization or training safety.

### Runtime environment inventory cycle

Test a read-only inventory of every local virtual-environment file, including
unrecorded files and bytecode, with stable file reads and symlink boundaries.
Bind the inventory to the interpreter and compare it against a draft runtime
manifest. Keep base Python libraries, operating-system libraries and process
attestation as separate unresolved dependencies.

### Generation accounting cycle

Test that collection records the stage budget counters before and after the
scheduled slots, reports their deltas, and reconciles captured call and token
reservations against those deltas. Missing or conflicting accounting must fail
the replay; sandbox counts remain local journal evidence, not authenticated
execution traces.

### Comparison accounting cycle

Test that complete comparisons freeze their shared budget counters, preserve
interrupted reservations, and reconcile all captured model-call and token
reservations against the frozen deltas. Sandbox counts remain local journal
evidence; missing or inconsistent accounting must prevent a capture replay.

### Collection server contract cycle

Test collection transport against the declared guarded model, parent adapter,
thinking canary, ledger identity and 200,000-token ceiling. Require a local
pre-dispatch reservation for each request and reject ledger drift or excess
server reservations. Server claims remain distinct from authenticated process
and model identity and do not authorize execution.

### Comparison reservation binding cycle

Test that the guarded comparison transport is bound to its stage budget and
requires an exact local reservation before consulting either server. Reject
unreserved, duplicate, mismatched and over-budget dispatches while preserving
server-contract and adapter checks. These checks remain local protocol evidence.

### Historical duplicate reference cycle

Test building duplicate-only fingerprints from bounded historical source
records and the registered private probes, with ancestry confined to the
generation partition and explicit sandbox-start limits. Reject missing ancestry,
source-hash conflicts, probe failures and development overlap rather than
silently omitting references. Caller-supplied source origins and history
completeness remain unverified; reference records never become training data.

### Exact duplicate feedback cycle

Test bounded repairs after exact source or AST duplication, using only the
model's own previous source and a generic category. Replay each attempt against
the frozen reference pool. Keep sampled private-behavior matches terminal so
private probe-derived feedback is not sent to the model.

### Collection attempt accounting cycle

Test that the generation report counts every attempt status, including exact
duplicates followed by repairs, alongside final slot outcomes. Replay must
recompute the attempt counts and reject conflicting summaries.

### Quarantined MLX dataset cycle

Test materializing the checked chat rows into an owner-only, non-overwritable
JSONL dataset and an audit record bound to the preparation and dataset hashes.
A failed or interrupted write must not be silently reused. This artifact is
quarantined and cannot bypass provenance, preflight, trainer or authorization
gates.

### MLX dataset loader audit cycle

Test loading the quarantined artifact through an injected MLX-compatible local
dataset loader after readback verification. Check every loaded row and masked
token span against the prepared records, require empty validation and test
splits, and reject loader or masking discrepancies. Exercise the pinned local
MLX loader with synthetic data and the local tokenizer without loading model
weights or invoking training. This is format evidence, not provenance or
authorization.
