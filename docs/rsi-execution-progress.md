# Issue 2 execution status

The [research design](rsi-research-protocol.md) and
[v2 target amendment](../research/rsi/design-v2.md) remain prospective. This
branch is an offline implementation in progress, not an executed experiment.

## Process monitoring

`src.arc.rsi_process_monitor.watch_owned_process` watches an already-created,
unreaped child that leads its own session. It checks resource samples and sends
termination signals only to that owned process group, retaining the leader PID
until cleanup. Missing, stale, or unavailable samples stop the child. On 64-bit
macOS it supports Python builds lacking `os.waitid` through the Darwin libc
interface. `src.arc.rsi_process_sampling` supplies host and process-tree samples.

Local verification used short-lived Python fixture children, including one that
ignored `SIGTERM`; no model or training process was started. In a local fixture
using the real host sampler, the leader exited with code 0 and three samples
were observed. macOS returned a permission uncertainty during group cleanup, so
`group_signals_confirmed` was false. The monitor does not prove that descendants
which escape its session are gone, that samples capture instantaneous peaks, or
that a model trained successfully. Training integration and independent process,
resource, artifact and adapter verification remain unfinished.

## Current gates

- The slot lifecycle consumes a claimed slot before fixture execution. A
  missing completion remains uncertain and cannot be replayed automatically.
  The fixture journal checks recorded byte hashes but is not an authenticated
  run seal and cannot establish training eligibility.
- The task split remains provisional and its retained-history inventory cannot
  prove the absence of missing prompt exposure or model pretraining exposure.
- Draft registration, replay, transport consistency and SFT prechecks do not
  establish an independently authenticated run or authorize training.
- A guarded live collector, independently audited run evidence, training gate,
  monitor integration and matched evaluation are not yet complete.
- Live model calls, training, external compute and independent scoring require
  their separately scoped authorizations and frozen budgets. None were run here.

### Scheduled call reservations

Owner-only, exclusive markers now consume each generation/slot/attempt before
fixture processing. The schedule permits at most 30 generation requests per
generation; a marker survives interruption and cannot be reused. The fixture
journal exercises this reservation but performs no model calls. These markers
are not an authenticated ledger: they do not account for calibration calls,
reserved tokens, sandbox starts or wall time, and a writer with access to the
run directory can alter them. Coordinated completion and dispatch, a shared
stage budget, independent provenance audit, and live authorization remain
required before any live collector can rely on them.

### Shared stage budget

A local, lock-serialized budget now reserves generation and canary calls against
one per-generation ceiling of 32 calls and 200,000 tokens, and reserves sandbox
starts against the 300-start ceiling. Reservations survive process interruption;
wall and monotonic clocks enforce the 3,600-second window and fail closed on
observed rollback. Adapters connect the budget to the capturing backend and
sandbox interface. The caller must supply the remaining-time callback and use
these adapters for every dispatch. These controls depend on cooperating local
processes and mutable local files; they do not authenticate a ledger, provide
hard termination of an already running sandbox, or authorize a live experiment.
No full collector or independent run audit is claimed by these components.
