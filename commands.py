import glob
import logging
import os
import warnings
from typing import Any, Dict

import fire
import torch
from hydra import compose, initialize

from ag_news_classifier.infer import infer
from ag_news_classifier.train import train, validate


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*_register_pytree_node.*")
warnings.filterwarnings("ignore", message=".*torch.utils._pytree.*")

logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("datasets").setLevel(logging.ERROR)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

torch.set_float32_matmul_precision("medium")


def get_latest_checkpoint() -> str:
    checkpoint_pattern = "lightning_logs/version_*/checkpoints/*.ckpt"
    checkpoints = glob.glob(checkpoint_pattern)
    if not checkpoints:
        raise FileNotFoundError("No checkpoint files found")
    return max(checkpoints, key=os.path.getctime)


class CLI:
    def __init__(self, config_name: str = "config", config_path: str = "conf"):
        with initialize(version_base=None, config_path=config_path):
            self.cfg = compose(config_name=config_name)

    def train(self) -> Dict[str, Any]:
        train_results = train(self.cfg)
        print(f"Training results: {train_results}")

        try:
            checkpoint_path = get_latest_checkpoint()
            val_results = validate(checkpoint_path, self.cfg)
            print(f"Validation results: {val_results}")

            return {
                "train": train_results,
                "val": val_results,
            }
        except FileNotFoundError:
            return {"train": train_results}

    def infer(self, text: str, checkpoint: str = None) -> Dict[str, Any]:
        if checkpoint is None:
            checkpoint = get_latest_checkpoint()

        results = infer(checkpoint, text)
        print(f"Inference results: {results}")
        return results


if __name__ == "__main__":
    fire.Fire(CLI)
