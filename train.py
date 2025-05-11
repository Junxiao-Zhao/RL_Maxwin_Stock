import os
import logging
from datetime import datetime

import hydra
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from datasets import load_dataset
from transformers import (
    PatchTSTForClassification,
    PatchTSTConfig,
    EarlyStoppingCallback,
    set_seed,
)
from trl.trl import GRPOConfig

from src import (
    ActionGRPOTrainer,
    compute_yield_policy1,
)

os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"


@hydra.main(version_base=None, config_path="config", config_name="train")
def train(cfg: DictConfig) -> float:

    logger = logging.getLogger(__name__)
    cfg = cfg.model
    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
    hydra_cfg = HydraConfig.get()
    logger.info(cfg)
    logger.info(hydra_cfg.overrides.task)

    set_seed(cfg['seed'])

    # Model
    config = PatchTSTConfig(**resolved_cfg["PatchTSTConfig"])
    model = PatchTSTForClassification(config).to(cfg["device"])

    # Data
    train_ds = load_dataset("parquet", data_files=cfg["dataset"]["train_fp"], split="train")
    eval_ds = load_dataset("parquet", data_files=cfg["dataset"]["eval_fp"], split="train")

    # Prepare Trainer
    cfg['GRPOConfig']['output_dir'] += "/" + datetime.now().strftime("%Y%m%d_%H%M%S")
    training_args = GRPOConfig(**cfg['GRPOConfig'])

    trainer = ActionGRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        reward_funcs=compute_yield_policy1,
        # callbacks=[
        #     EarlyStoppingCallback(**cfg['EarlyStoppingCallback']),
        # ],
    )

    # Train & Evaluate
    trainer.train(**cfg["train"])

    return trainer.state.best_metric


if __name__ == '__main__':
    try:
        train()
    except Exception as e:
        logging.getLogger(__name__).exception(e)
