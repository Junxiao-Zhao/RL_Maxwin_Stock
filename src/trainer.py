from typing import Any, Union
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from accelerate.utils.operations import gather
from trl import GRPOTrainer
from trl.extras.profiling import profiling_context


class ActionGRPOTrainer(GRPOTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reward_metrics = {
            "train": defaultdict(list),
            "eval": defaultdict(list)
        }

    def _evaluate(self, trial, ignore_keys_for_eval, skip_scheduler=False):
        metrics = super()._evaluate(trial, ignore_keys_for_eval,
                                    skip_scheduler)
        trl_metrics = {
            f"eval_{k}": np.mean(v)
            for k, v in self.reward_metrics["eval"].items()
        }
        metrics.update(trl_metrics)
        self.reward_metrics["train"].clear()
        self.reward_metrics["eval"].clear()

        return metrics

    def _get_per_token_logps(self,
                             model,
                             input_ids,
                             attention_mask,
                             logits_to_keep,
                             return_actions=False):
        logits = model(input_ids=input_ids,
                       attention_mask=attention_mask,
                       logits_to_keep=logits_to_keep + 1).logits
        logits = logits[:, :-1, :]
        logits = logits[:, -logits_to_keep:]
        logits = logits / self.temperature

        logits = F.softmax(logits, dim=-1)
        dist = Categorical(logits)
        actions = dist.sample(torch.Size([self.num_generations]))
        log_probs = dist.log_prob(actions)  # (N, B, S)

        # (B * N, S)
        actions = actions.permute(1, 0, 2).reshape(-1, logits_to_keep)
        log_probs = log_probs.permute(1, 0, 2).reshape(-1, logits_to_keep)

        if return_actions:
            return log_probs, actions

        return log_probs

    def _generate_and_score_completions(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device

        prompt_ids = torch.tensor(
            [x["prompt_ids"] for x in inputs[::self.num_generations]],
            dtype=torch.float32)
        prompt_mask = torch.tensor(
            [x["prompt_mask"] for x in inputs[::self.num_generations]],
            dtype=torch.float32)
        completion_ids = torch.tensor(
            [x["completion_ids"] for x in inputs[::self.num_generations]],
            dtype=torch.float32)
        completion_mask = torch.tensor(
            [x["completion_mask"] for x in inputs[::self.num_generations]],
            dtype=torch.float32)
        prompt_ids = super(GRPOTrainer, self)._prepare_inputs(prompt_ids)
        prompt_mask = super(GRPOTrainer, self)._prepare_inputs(prompt_mask)
        completion_ids = super(GRPOTrainer,
                               self)._prepare_inputs(completion_ids)
        completion_mask = super(GRPOTrainer,
                                self)._prepare_inputs(completion_mask)

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length:]
            prompt_mask = prompt_mask[:, -self.max_prompt_length:]

        prompt_completion_ids = torch.concat([prompt_ids, completion_ids],
                                             dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        with torch.no_grad():
            # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's
            # computation here, and use per_token_logps.detach() instead.
            old_per_token_logps, actions = self._get_per_token_logps(
                self.model,
                prompt_completion_ids,
                attention_mask,
                logits_to_keep,
                return_actions=True)
            if self.num_iterations <= 1:
                old_per_token_logps = None

            if self.beta == 0.0:
                ref_per_token_logps = None
            elif self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask,
                    logits_to_keep)
            else:
                with self.accelerator.unwrap_model(
                        self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model, prompt_completion_ids, attention_mask,
                        logits_to_keep)

        rewards_per_func = torch.zeros(len(actions),
                                       len(self.reward_funcs),
                                       device=device)  # (B * N, 1)
        for i, (reward_func, reward_processing_class) in enumerate(
                zip(self.reward_funcs, self.reward_processing_classes)):
            if isinstance(
                    reward_func, nn.Module
            ):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = f"reward {reward_func.config._name_or_path.split('/')[-1]}"
            else:
                reward_func_name = reward_func.__name__
            with profiling_context(self, reward_func_name):
                if isinstance(
                        reward_func, nn.Module
                ):  # Module instead of PretrainedModel for compat with compiled models
                    raise NotImplementedError()
                else:
                    # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                    keys = [
                        key for key in inputs[0]
                        if not key.startswith("prompt")
                    ]
                    reward_kwargs = {
                        key: [example[key] for example in inputs]
                        for key in keys
                    }
                    output_reward_func = reward_func(
                        actions=actions,
                        **reward_kwargs,
                    )
                    # Convert None values to NaN
                    output_reward_func = [
                        reward if reward is not None else torch.nan
                        for reward in output_reward_func
                    ]

                    rewards_per_func[:, i] = torch.tensor(output_reward_func,
                                                          dtype=torch.float32,
                                                          device=device)

        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes
        rewards_per_func = gather(rewards_per_func)  # (B * N, 1)

        # Apply weights to each reward function's output and sum
        rewards = (rewards_per_func *
                   self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1,
                                            self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(
            self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(
            self.num_generations, dim=0)
        advantages = rewards - mean_grouped_rewards
        if self.args.scale_rewards:
            advantages = advantages / (std_grouped_rewards + 1e-4)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(actions),
            (self.accelerator.process_index + 1) * len(actions),
        )
        advantages = advantages[process_slice]

        # Log the metrics
        mode = "eval" if self.control.should_evaluate else "train"

        if mode == "train":
            self._total_train_tokens += self.accelerator.gather_for_metrics(
                attention_mask.sum()).sum().item()
        self._metrics[mode]["num_tokens"] = [self._total_train_tokens]

        completion_length = self.accelerator.gather_for_metrics(
            completion_mask.sum(1)).float().mean().item()
        self._metrics[mode]["completion_length"].append(completion_length)

        # Calculate mean reward per function, but only for samples where the function was applied
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(
                    reward_func, nn.Module
            ):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = reward_func.config._name_or_path.split(
                    "/")[-1]
            else:
                reward_func_name = reward_func.__name__
            # Only calculate mean for samples where this reward function was applied (non-NaN values)
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}"].append(
                mean_rewards)
        self._metrics[mode]["reward"].append(rewards.mean().item())
        self._metrics[mode]["reward_std"].append(
            std_grouped_rewards.mean().item())
        self.reward_metrics[mode]["reward"].append(rewards.mean().item())

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
        }
