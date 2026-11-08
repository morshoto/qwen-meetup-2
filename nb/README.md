# Qwen × ARC notebooks

The notebooks are grouped by purpose:

- [Results and LT material](results/): reports and visualizations of recorded experiments. Start with the [overview](results/00_overview.ipynb) or the [Qwen3.8 A100 development pilot](results/14_qwen38_a100_results.ipynb). The latter embeds a fixed snapshot; rerunning it does not read new experiment results.
- [Experiments](experiments/): notebooks for preparing or running work. The [Qwen3.8 Phase A notebook](experiments/002_qwen38_phase_a_colab.ipynb) adds pilot diagnostics and a development gate to the [original Colab notebook](experiments/001_qwen38_colab_arc.ipynb). The prior pilot had no valid candidates; a reviewed format fix and successful retry are still needed before Phase A. Both notebooks require a compatible Colab GPU, Google Drive, inputs and a commit containing their supporting code; see [PR #15](https://github.com/morshoto/qwen-meetup-2/pull/15). They do not train or load an RSI adapter. The separate [data template](experiments/data_001.ipynb) is a starting point for new work.

The numbered E1–E9 results notebooks embed aggregate results recorded through November 5, 2026 and link to their sources. Some detailed artifact links resolve only when the locally retained, Git-ignored artifacts are present; these reports do not read them. The saved charts can be viewed without running code. The [50-task RSI comparison notebook](results/13_rsi_exploratory_50.ipynb) reads existing results or an explicitly selected interrupted-run record; it does not train or run a model.

To rerun the static reports, install the project environment with `uv sync`, open them in a Jupyter-compatible editor, and run all cells. They use pandas and Matplotlib and do not need a model server, GPU, Docker, or network access.
