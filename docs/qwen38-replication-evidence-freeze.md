# Qwen3.8 replication evidence freeze: initial inventory

The #16 code revision being replicated is `15ba1ad007ee907c1a7923c8371af0e696f09b7a` (PR #17). The model revision is `10c35caafbb80f7dc6a7a432cdd11af10a6d4818`; the adapter SHA-256 is `b363fd2bd15114577d96141b2e03500e7d3d241a4b4a64b6176f25dbb46ac9f6`. The completed exploratory report has content digest `023ac3d4afa8982d43a24e2b328f4471e127a7c2f9a24109d582d9e89d193b70`.

A private, exclusive inventory of the original #16 run tree was created under `artifacts/qwen38-replication-v1/source-inventory.json`. It contains 3,087 file hashes, including the collection, training and recovered evaluation journals and worker evidence; its content digest is `ea309257ef1e10e081a68ec6878b22b90ef408806c09f0279989d6095483d16e`. Before inventorying, the saved SFT manifest, training result, adapter, adapter configuration, evaluation descriptor, termination receipt and report were checked against their pinned file hashes. The inventory records source bytes without modifying them and contains no response or answer grids.

This is an **initial inventory**, not a complete independent replay of all sources and exposure evidence or a new-task registration. It does not establish that the reserved 25 parents are untouched, and it does not authorize inference. The complete protocol, selected IDs and budget must be frozen before new predictions or inspection of target answers. Unknown semantic and pretraining exposure remain limitations.

On 2026-11-10, a read-only verification recomputed the execution descriptor's
content digest (`08e4c2cabaca7f331c43a5bb12ddf5e1924ac68e573e3819e1e685044cf1038b`),
all 34 recorded execution source-file hashes, all 15 model snapshot file
hashes (including the tokenizer and chat template), and the 3,087 run-artifact
hashes against the initial inventory. Installed package versions matched the
execution descriptor. The model's Hugging Face snapshot files resolve through
the local cache's blob links; the file contents were hashed. This check did not
read target solutions or validate whether all historical exposure is known.

An answer-free visible-input audit checked the 25 previously reserved
validation parents against the proposal's exclusions, all 40 actual fresh
generation parents, the 203 distinct historical FP8 Phase A/development
parents, the 14 saved SFT parents, and the 120 official evaluation challenges.
It propagated shared visible inputs and recorded lineage transitively and
found no selected overlap; 975 Training parents were in the resulting blocked
set. The proposal, challenge and historical-manifest identities and selected
challenge fingerprints were checked. The saved lineage register is empty and
the lineage register cannot prove unseen tasks or authorize a model run.

An additional bounded answer-fingerprint check found zero matches between
the 25 reserved fingerprints already saved in the #16 proposal and 975
blocked Training parents, and no duplicate fingerprints within the 25.
The Training solutions file was hash-checked against the prior registration;
the selective scanner decoded only blocked-parent answers and skipped the
reserved values without decoding or retaining them. The scanner necessarily
reads the file bytes to verify its hash and navigate its structure. This
checks exact equality only; unrecorded lineage and semantic or pretraining
exposure remain unknown. Full source replay and the final preregistration are
still required before inference.

The bounded check was subsequently replayed by
`src/arc/qwen38_replication_audit.py`, including verification of the original
run inventory, the pinned historical and SFT manifests, each of 294 recorded
fresh starts, the input identities and both visible and answer-fingerprint
exclusions. Its private receipt is
`artifacts/qwen38-replication-v1/bounded-audit.json` with content digest
`572ce6a11cf545b8245de098c235ef3a81b38d5ef0b3c97484d0759597246b3a`.
The receipt records a bounded exploratory audit, not proof of semantic
independence or authorization to dispatch a worker.
