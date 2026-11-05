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
