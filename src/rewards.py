from typing import Sequence

import torch


def compute_yield_policy1(
    actions: torch.Tensor,  # (B * N, S)
    completion_mask: torch.Tensor,  # (B * N, S)
    open_price: Sequence,  # (B * N, S)
    close_price: Sequence,  # (B * N, S)
    slippage: float = 0.01,
    stamps: float = 0.0,
    service_fee: float = 1.3e-4,
    assets: float = 2e5,
    **kwargs,
):

    batch_n, seq_len = actions.shape
    zero = torch.zeros((batch_n, 1), device=actions.device)
    if not isinstance(completion_mask, torch.Tensor):
        completion_mask = torch.tensor(completion_mask, dtype=torch.float32)
    completion_mask = completion_mask.to(actions.device)
    actions.masked_fill_(completion_mask == 0, 0)
    diff = torch.diff(actions, prepend=zero, append=zero)
    yields = torch.zeros(batch_n).to(actions.device)

    for i in range(batch_n):
        total_assets = assets

        # calculate the holding periods
        start = torch.where(diff[i] == 1)[0]
        end = torch.where(diff[i] == -1)[0]

        for span in list(zip(start.tolist(), end.tolist())):
            start_idx, end_idx = span

            bid_rate = (open_price[i][start_idx] + slippage) * (1 + service_fee)
            shares = total_assets // bid_rate
            total_assets = total_assets % bid_rate

            if end_idx < seq_len and completion_mask[i, end_idx] == 1:
                ask_rate = (open_price[i][end_idx] - slippage) * (1 - service_fee - stamps)
            else:
                ask_rate = close_price[i][end_idx - 1]

            total_assets += shares * ask_rate

        yields[i] = (total_assets - assets) / assets

    return yields
