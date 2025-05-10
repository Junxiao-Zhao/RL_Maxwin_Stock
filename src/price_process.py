from typing import List

import torch
import torch.nn.functional as F
import pandas as pd
from datasets import Dataset


class PriceProcessor:

    def __init__(
        self,
        input_cols: List[str],
        reward_cols: List[str],
        date_col: str,
        prompt_len: int = 32,
        completion_len: int = 32,
    ):
        self.cols = list(set(input_cols + reward_cols))
        self.input_cols = input_cols
        self.reward_cols = reward_cols
        self.date_col = date_col
        self.prompt_len = prompt_len
        self.max_len = prompt_len + completion_len

    def __call__(self, window_df: pd.DataFrame):

        window_df = window_df.iloc[:self.max_len]
        # window_df[self.date_col] = window_df[self.date_col].apply(
        #     lambda x: x.timestamp())

        input_ids = torch.tensor(window_df[self.input_cols].values, dtype=torch.float32)

        # pad
        prefix_fill = max(0, self.prompt_len - input_ids.size(0))
        suffix_fill = max(0, self.max_len - input_ids.size(0) - prefix_fill)
        pad_dim = (0, 0, prefix_fill, suffix_fill)
        input_ids = F.pad(input_ids, pad_dim, value=0)

        # mask
        attention_mask = torch.ones(input_ids.shape[:-1], dtype=torch.float32)
        if prefix_fill:
            attention_mask[:prefix_fill] = 0
        if suffix_fill:
            attention_mask[-suffix_fill:] = 0

        # split
        prompt_ids, prompt_mask = input_ids[:self.prompt_len], attention_mask[:self.prompt_len]
        completion_ids, completion_mask = input_ids[self.prompt_len:], attention_mask[self.prompt_len:]

        result = {
            # "trade_date": window_df[self.date_col].values,
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
        }

        # prepare inputs from reward_func
        window_dict = window_df[self.reward_cols].to_dict(orient="list")
        window_dict = {
            k: F.pad(torch.tensor(v, dtype=torch.float32), pad_dim[2:], value=0)[self.prompt_len:]
            for k, v in window_dict.items()
        }
        result.update(window_dict)

        return result

    def rolling(self, df: pd.DataFrame, **kwargs):

        df = df[[self.date_col] + self.cols]
        df.loc[:, self.date_col] = pd.to_datetime(df[self.date_col])
        df = df.sort_values(by=self.date_col, ignore_index=True).bfill().ffill()

        windows = list(df.rolling(**kwargs))
        windows = list(filter(lambda x: len(x) >= self.prompt_len, windows))

        results = list(map(self, windows))
        dataset = Dataset.from_list(results)

        return dataset
