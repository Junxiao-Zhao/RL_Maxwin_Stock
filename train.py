import os
import logging
from datetime import datetime

import hydra
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from datasets import concatenate_datasets, DatasetDict
from transformers import (
    AutoConfig,
    EarlyStoppingCallback,
    set_seed,
)
from trl import GRPOConfig

from src import (
    PriceProcessor,
    Qwen2ForAction,
    ActionGRPOTrainer,
    compute_yield_policy1,
)

os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"


@hydra.main(version_base=None, config_path="config", config_name="train")
def train(cfg: DictConfig) -> float:

    logger = logging.getLogger(__name__)
    cfg = OmegaConf.to_container(cfg, resolve=True)['model']
    hydra_cfg = HydraConfig.get()
    logger.info(cfg)
    logger.info(hydra_cfg.overrides.task)

    set_seed(cfg['seed'])

    # Model
    config = AutoConfig.from_pretrained(**cfg['llm_config']['from_pretrained'])
    config.update(cfg['config'])
    if cfg['load_pretrained']:
        model = Qwen2ForAction.from_pretrained(**cfg['llm']['from_pretrained'], config=config)
    else:
        model = Qwen2ForAction(config)
    model = model.to(cfg["device"])

    # Prepare data
    overall_df = pd.read_csv(cfg["csv_path"])
    train_df, eval_df = overall_df.iloc[:-500], overall_df.iloc[-500:]
    processor = PriceProcessor(**cfg["PriceProcessor"]["init"])

    max_len = cfg["prompt_len"] + cfg["completion_len"]
    dataset = DatasetDict()

    for split, df in zip(["train", "test"], [train_df, eval_df]):
        dataset_ls = []
        for step in cfg["PriceProcessor"]["rolling"]["steps"]:
            for window in cfg["PriceProcessor"]["rolling"]["windows"]:
                if window > max_len:
                    continue
                dataset_ls.append(processor.rolling(df, window=window, step=step))
        dataset[split] = concatenate_datasets(dataset_ls)
    ds_tr_eval = dataset.rename_columns({"open": "open_price", "close": "close_price"})

    # ds_tr_eval = dataset.train_test_split(**cfg['dataset']['train_test_split'])
    logger.info(ds_tr_eval)

    # Prepare Trainer
    # trial_num = hydra_cfg.job.get("id", cfg["trial_name"])
    cfg['GRPOConfig']['output_dir'] += "/" + datetime.now().strftime("%Y%m%d_%H%M%S")
    training_args = GRPOConfig(**cfg['GRPOConfig'])

    trainer = ActionGRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=ds_tr_eval['train'],
        eval_dataset=ds_tr_eval['test'],
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
