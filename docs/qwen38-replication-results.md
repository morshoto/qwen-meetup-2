# Frozen LoRA replication results

The frozen base and adapter each solved **4 of 25 tasks** with either of two recorded candidates. This separate exploratory set did not reproduce the earlier 50-task either-candidate increase from 2/50 to 5/50. The sets differ, so their scores do not isolate a cause or establish generalization or recursive self-improvement.

| Outcome | Frozen base | Frozen adapter |
| --- | ---: | ---: |
| First candidate correct | 2/25 | 3/25 |
| Either candidate correct | 4/25 | 4/25 |
| Valid candidates | 39/50 | 50/50 |
| Duplicate raw candidate pairs | 3/25 | 2/25 |
| Sum of attempt runtimes | 2,175.491 s | 3,765.131 s |

All 25 pairs were complete: one task was gained, one lost, three correct in both arms, and 20 incorrect in both. The predeclared two-sided exact sign test gives **p = 1.0** for the two discordant pairs. There were no inference errors, missing tasks, or partial pairs. These are recorded attempts, not independent samples; the small number of tasks and discordances limits inference.

The completed run made 100 attempts, with 100 successful worker receipts and all 100 process groups gone. Total controller wall time was 7,611.129 seconds. Across receipt samples, minimum free memory was 50%, maximum swap use was 0 bytes, and maximum reported MLX peak allocation was 21.5855 GB. Sampling does not establish an instantaneous system-wide peak. The predeclared thresholds were 10% free memory and 16 GiB swap, with a 600-second attempt timeout and a 24-hour run limit.

The frozen `mlx-community/Qwen3.8-27B-4bit` revision was `10c35caafbb80f7dc6a7a432cdd11af10a6d4818`; the adapter SHA-256 was `b363fd2bd15114577d96141b2e03500e7d3d241a4b4a64b6176f25dbb46ac9f6`. The prior #16 report was `023ac3d4afa8982d43a24e2b328f4471e127a7c2f9a24109d582d9e89d193b70`. The mixed-source adapter used 2 fresh 4-bit and 12 historical FP8 unique parents; nine length exclusions were recorded. No new training or tuning was performed for this replication.

The [evidence freeze](qwen38-replication-evidence-freeze.md) describes the recorded exposure, visible-input, answer-fingerprint, and lineage checks against 975 blocked records and a 3,087-file private inventory. None of the 25 reserved tasks matched the recorded exclusions or fingerprints. The earlier allocation had already computed the 25 answer fingerprints; the lineage register was empty, and unknown semantic or pretraining exposure cannot be ruled out. Consequently this is exploratory, not proof of a completely unexposed holdout. Official evaluation challenges were used only for exclusion checks; official evaluation solutions were not accessed.

The [registration](../research/rsi/qwen38-replication-v1.json) froze settings and source identities before new predictions or inspection of target answers. Each arm used temperature 0.6, two candidates, 4,096 maximum completion tokens, an 18,432-token context, and matched seeds; the base ran first. See the [protocol](qwen38-replication-protocol.md) and [results notebook](../nb/results/18_qwen38_frozen_replication.ipynb).

Private receipt SHA-256 identities: registration `bdcea4df4019eed39b73a786a8e07dec04410e1d37660e1b54992d14a2682e2e`; bounded audit `572ce6a11cf545b8245de098c235ef3a81b38d5ef0b3c97484d0759597246b3a`; terminal receipt `665af9037584ee28a3107c33109590a965128d23a0a05f6f2aac7decdae2e803`; journal `30b6fca90507dc9d4caa5e75c0f425557b6d2355ef0af3900b45d8ea1d68bd69`; offline report `bc6a45edb4850feac1f658707b215bb3e66cbd4b5e7b12ebd4bd5636b1690c4d`. The report was produced only after terminal receipt and journal replay; private raw candidates and answer grids are not included here.
