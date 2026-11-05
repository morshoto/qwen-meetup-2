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
