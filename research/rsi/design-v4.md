# Qwen RSI research design v4: separate technical canaries

This prospective amendment records the user's updated strategy. It retains the
[v3 bounded exposure review](design-v3.md), the same-model strict training
gate, and the existing provenance, resource and authorization requirements.
It is a design record, not permission to execute a canary or train a model.

## Separate task partitions

Reserve the proposed 50 exploratory tasks from generation, leaving 750
generation tasks and the prior 200 development tasks. The 50 and 200 are
separate and neither is used for generation or training. Before execution,
independently review the history and the [allocation checks](audits/allocation-review.json),
freeze the manifests and all code/data hashes, and preserve the uncertainty in
unknown historical project exposure and model pretraining. Review the 50-task
results before deciding whether to run the 200-task comparison.

## Technical canaries before the Oracle gate

A one-step, synthetic-data-only training and adapter save/reload canary may be
registered and authorized **before** five new Oracles exist. The canary is a
technical pipeline check. Its fixed fabricated data must be separate from all
ARC task inputs and outputs available to the review; it is never generation
training data. Its temporary adapter is quarantined and cannot be promoted,
used to generate candidates, used as an RSI parent or used for evaluation.
Passing it does not establish RSI improvement, data quality, or benchmark
competitiveness.

Treat 14B and 32B as separate attempts with separate model identity and local
file hashes, runtime and tokenizer manifests, resource baselines, budgets,
process supervision, output directories, authorization, and independently
verified save/reload observations. A successful 14B canary cannot authorize or
establish the feasibility of 32B training. Stop each canary on OOM, missing or
failed monitoring, non-finite loss, invalid reload, or any registered resource
limit. Register safe per-model limits and review machine state before execution;
no value in this design is an assertion that 32B fits on the current Mac.

## Oracle generation and training

The corrected E8 rerun contains one previously audited same-model candidate for
each model, retained separately as a duplicate reference. The original E8
Full run has an unresolved dependency-registration gap. Do not count either
historical candidate toward the current new-only five-Oracle gate. Whether a
future protocol may count one historical candidate remains undecided and must
be explicitly registered before affected collection or training.

Use the recorded E8/E9 failure categories to design a separately registered
Oracle-generation method. Public repair feedback may describe permitted generic
syntax, contract, safety, runtime, visible mismatch or exact duplicate errors;
never reveal private probe outputs or correct answers. Register all attempt,
model-call, token, sandbox and time limits before dispatch. Report new, duplicate
and invalid candidates separately. Stop RSI training for a model unless five
distinct new same-model Oracles pass its independent provenance and data audit.
The earlier synthetic canary cannot satisfy this gate.
