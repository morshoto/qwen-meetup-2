# Qwen × ARC experiment notebooks

Start with [the overview](00_overview.ipynb), then follow the notebooks in order:

1. [P3-B: collection and adapter training](01_p3b_training.ipynb)
2. [E1: failure diagnosis](02_e1_diagnosis.ipynb)
3. [E2: search strategy comparison](03_e2_search.ipynb)
4. [E3: solution collection](04_e3_collection.ipynb)
5. [E4: synthetic transformation search](05_e4_operators.ipynb)
6. [E5: autonomous self-play](06_e5_selfplay.ipynb)
7. [E6: generator distillation](07_e6_distillation.ipynb)
8. [E6-F: Oracle repair](08_e6f_repair.ipynb)
9. [E6-G: novelty feedback](09_e6g_novelty.ipynb)
10. [E7: paired novelty comparison](10_e7_paired.ipynb)
11. [E8: model scaling](11_e8_scaling.ipynb)
12. [E9: model-isolated collection](12_e9_collection.ipynb)

Each notebook embeds aggregate results recorded through November 5, 2026 and links to its sources. Some detailed artifact links resolve only when the locally retained, Git-ignored artifacts are present; the notebooks themselves do not read them. The saved charts can be viewed without running code.

To rerun the notebooks, install the project environment with `uv sync`, open the notebooks with a Jupyter-compatible editor using the project Python environment, and run all cells. They use pandas and Matplotlib; no model server, GPU, Docker, or network access is needed for execution. The existing `data_001.ipynb` is a separate template.

## Colab ARC pilot and Phase A handoff

[Qwen3.8 ARC Colab](14_qwen38_colab_arc.ipynb) is an operational notebook, separate from the precomputed narratives above. Import it into Colab, select an A100 GPU, and enter the reviewed repository commit SHA. Its default FP8 checkpoint and 18,432-token context differ from the original BF16/MLX configurations. It mounts Google Drive for public challenge inputs and task journals; do not place held-out solutions in the mounted account during inference. Keep the full run gated until the development pilot and registration have been reviewed.

For Phase A registration, copy the notebook-generated `phase-a-model-config.json` to a local checkout of the same commit with the registered challenges and solutions under `data/raw/`. Run `python -m src.cli.lt_single prepare --config /path/to/phase-a-model-config.json --directory /path/to/new-run`. Transfer only `manifest.json`, `export.json`, and `receipt.json` to a new Drive run directory. Copy the completed journal back to that same local checkout and use `python -m src.cli.lt_single score --directory /path/to/new-run`; never transfer solutions for inference. Model and prompt identity, context capacity, format validity, interruptions, and provider usage must be reviewed independently. Budget calculations are estimates, not a provider-enforced limit, and an idle connected runtime continues to consume units.
