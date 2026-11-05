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
