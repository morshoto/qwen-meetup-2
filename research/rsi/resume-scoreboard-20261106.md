# RSI resume scoreboard

Status at this checkpoint: `goal_not_achieved_with_evidence`; neither model is
approved to launch. This is a checkpoint, not a claim that the broader research
goal has been achieved.

| Model | Revision | Review and safety | New strict Oracles in this resume | Canary | Gen0 to Gen1 | Adapter SHA | Gen0 / Gen1 on 50 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 14B | `a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4` | Separate-context review incomplete; cleanup and data isolation unverified | Not collected | Not run | Not run | None | Not run |
| 32B | `bcaaf7f538adf166c1080a2befdb4f6019f66639` | Separate-context review incomplete; cleanup and data isolation unverified | Not collected | Not run | Not run | None | Not run |

No new Oracle collection, real-model optimizer step, 50-task evaluation, or
external paid service was used in this resume. This is not an observed failure
to obtain five Oracles; the search did not start. Historical E9 records report
zero new strict candidates for both models and one older candidate per model,
but the [separate-context review](resume-independent-context-review-20261106.md)
confirmed only retained-record consistency, not current training eligibility.

The [decision log](end-to-end-decision-log-20261106.md) records the separate
split review, local model-file hash comparisons, synthetic fixture results,
known cleanup limitations, and the provisional registrations. The latest
source-bound packet is local, owner-only `artifacts/rsi-resume-registration-20261106-v10/`;
it is not an execution registration. Historical seals were not replaced.
The local review index is `artifacts/rsi-resume-v10-review-index.json`; raw
historical and private inputs remain separate and must be reviewed in place.

## Required conditions for execution

- Review and bind a complete current launch closure, including model, runtime,
  source, split, generation-only export, settings and budgets.
- Establish and review evaluation-data isolation across filesystem aliases,
  inherited descriptors, services and IPC for the actual generation process.
- Establish and independently review owned-process cleanup, including escaped
  descendants, process identities, ports and containers. The present observer
  can only record some processes; all cleanup claims remain unverified.
- Only after the applicable safety and review gates, run model-specific
  canaries and Oracle collection. Only models with at least five audited,
  distinct same-model Oracles may proceed to QLoRA and the paired 50 tasks.

Next external action if the local safety boundary cannot be established: an
independent reviewer should inspect the local review packet and findings and
specify an available isolated execution environment with verified cleanup.
