"""Use the pinned MLX-VLM training dataset path for SFT length and mask checks."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.arc.qwen38_local_runner import MODEL_REVISION


def trainer_token_count(messages, dataset_factory):
    """Count actual trainer input IDs and require a nonempty completion-only mask."""
    item = dataset_factory(messages)[0]
    ids = np.asarray(item['input_ids']); attention = np.asarray(item['attention_mask'])
    if 'completion_mask' not in item:
        raise ValueError('trainer did not provide a completion mask')
    mask = np.asarray(item['completion_mask'])
    if (ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] < 2 or attention.shape != ids.shape or
        mask.shape != ids.shape or not np.all(attention == 1) or not np.all((mask == 0) | (mask == 1)) or
        mask[0, 0] != 0 or not np.any(mask[0, 1:] == 1) or np.any(np.diff(mask[0].astype(np.int64)) < 0)):
        raise ValueError('unsafe trainer completion mask')
    return int(ids.shape[1])


def make_trainer_counter(model_path):
    """Load the local processor, never the model weights, using the pinned trainer."""
    model_path = Path(model_path).resolve()
    if not model_path.is_dir() or model_path.name != MODEL_REVISION:
        raise ValueError('expected pinned local model snapshot')
    from mlx_vlm.trainer.datasets import VisionDataset
    from mlx_vlm.utils import load_image_processor, load_processor
    config = json.loads((model_path / 'config.json').read_bytes())
    # Match mlx_vlm.utils.load: model-specific processor registration occurs
    # while resolving the image processor, including for text-only examples.
    image_processor = load_image_processor(model_path)
    processor = load_processor(model_path, True)
    if image_processor is not None:
        processor.image_processor = image_processor
    def count(messages):
        return trainer_token_count(messages, lambda value: VisionDataset(
            [{'messages': value}], config, processor, train_on_completions=True))
    return count
