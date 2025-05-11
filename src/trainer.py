from typing import Any, Union
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from accelerate.utils.operations import gather
from trl.trl import GRPOTrainer
from trl.trl.extras.profiling import profiling_context


class ActionGRPOTrainer(GRPOTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reward_metrics = {"train": defaultdict(list), "eval": defaultdict(list)}

    def _evaluate(self, trial, ignore_keys_for_eval, skip_scheduler=False):
        metrics = super()._evaluate(trial, ignore_keys_for_eval, skip_scheduler)
        trl_metrics = {f"eval_{k}": np.mean(v) for k, v in self.reward_metrics["eval"].items()}
        metrics.update(trl_metrics)
        self.reward_metrics["train"].clear()
        self.reward_metrics["eval"].clear()

        return metrics

    def _get_per_token_logps(self, logits, return_actions=False):
        logits = logits / self.temperature
        logits_to_keep = logits.shape[1]

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

    def _patchtst_generation(self, model: nn.Module, unfold_prompt_ids: torch.Tensor, unfold_prompt_mask: torch.Tensor):

        batch_size, num_actions = unfold_prompt_ids.shape[:2]
        unfold_prompt_ids = unfold_prompt_ids.flatten(end_dim=1)
        unfold_prompt_mask = unfold_prompt_mask.flatten(end_dim=1)

        output = model(past_values=unfold_prompt_ids, past_observed_mask=unfold_prompt_mask)
        completion_ids = output.prediction_logits.reshape(batch_size, num_actions, -1)  # (B, S, 2)
        completion_mask = torch.ones(completion_ids.shape[:-1], dtype=torch.bool, device=completion_ids.device)

        return completion_ids, completion_mask

    def _generate_and_score_completions(
            self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device

        prompt_ids = torch.tensor([x["prompt_ids"] for x in inputs[::self.num_generations]], dtype=torch.float32)
        prompt_mask = torch.tensor([x["prompt_mask"] for x in inputs[::self.num_generations]], dtype=torch.float32)
        prompt_ids = super(GRPOTrainer, self)._prepare_inputs(prompt_ids)  # (B, C, K)
        prompt_mask = super(GRPOTrainer, self)._prepare_inputs(prompt_mask)

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length:]
            prompt_mask = prompt_mask[:, -self.max_prompt_length:]

        unfold_prompt_ids = prompt_ids.unfold(1, 32, 1).permute(0, 1, 3, 2)  # (B, S, 32, K)
        unfold_prompt_mask = prompt_mask.unfold(1, 32, 1).permute(0, 1, 3, 2)

        completion_ids, completion_mask = self._patchtst_generation(self.model, unfold_prompt_ids, unfold_prompt_mask)
        attention_mask = torch.cat([prompt_mask[:, :, 0], completion_mask], dim=1)
        completion_mask = completion_mask.repeat_interleave(self.num_generations, dim=0)

        with torch.no_grad():
            # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's
            # computation here, and use per_token_logps.detach() instead.
            old_per_token_logps, actions = self._get_per_token_logps(completion_ids, return_actions=True)
            if self.num_iterations <= 1:
                old_per_token_logps = None

            if self.beta == 0.0:
                ref_per_token_logps = None
            elif self.ref_model is not None:
                ref_completion_ids, _ = self._patchtst_generation(self.ref_model, unfold_prompt_ids, unfold_prompt_mask)
                ref_per_token_logps = self._get_per_token_logps(ref_completion_ids)
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(completion_ids)

        rewards_per_func = torch.zeros(len(actions), len(self.reward_funcs), device=device)  # (B * N, 1)
        for i, (reward_func,
                reward_processing_class) in enumerate(zip(self.reward_funcs, self.reward_processing_classes)):
            if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = f"reward {reward_func.config._name_or_path.split('/')[-1]}"
            else:
                reward_func_name = reward_func.__name__
            with profiling_context(self, reward_func_name):
                if isinstance(reward_func,
                              nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                    raise NotImplemented
                else:
                    # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                    keys = [key for key in inputs[0] if not key.startswith("prompt")]
                    reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
                    output_reward_func = reward_func(actions=actions, completion_mask=completion_mask, **reward_kwargs)
                    # Convert None values to NaN
                    output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]

                    rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes
        rewards_per_func = gather(rewards_per_func)  # (B * N, 1)

        # Apply weights to each reward function's output and sum
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
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
            self._total_train_tokens += self.accelerator.gather_for_metrics(attention_mask.sum()).sum().item()
        self._metrics[mode]["num_tokens"] = [self._total_train_tokens]

        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics[mode]["completion_length"].append(completion_length)

        # Calculate mean reward per function, but only for samples where the function was applied
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            # Only calculate mean for samples where this reward function was applied (non-NaN values)
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}"].append(mean_rewards)
        self._metrics[mode]["reward"].append(rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())
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

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        # Compute the per-token log probabilities for the model

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]

        unfold_prompt_ids = prompt_ids.unfold(1, 32, 1).permute(0, 1, 3, 2)  # (B, S, 32, K)
        unfold_prompt_mask = prompt_mask.unfold(1, 32, 1).permute(0, 1, 3, 2)

        completion_ids, completion_mask = self._patchtst_generation(self.model, unfold_prompt_ids, unfold_prompt_mask)
        per_token_logps = self._get_per_token_logps(completion_ids)
        completion_mask = completion_mask.repeat_interleave(self.num_generations, dim=0)

        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) -
                            1)

        # Compute the loss
        advantages = inputs["advantages"]
        # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's computation (see
        # _generate_and_score_completions) and use per_token_logps.detach() instead.
        old_per_token_logps = inputs["old_per_token_logps"] if self.num_iterations > 1 else per_token_logps.detach()
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl
        loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()

        # Log the metrics
        mode = "eval" if self.control.should_evaluate else "train"

        if self.beta != 0.0:
            mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
            self._metrics[mode]["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        is_clipped = (per_token_loss1 < per_token_loss2).float()
        clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum()
        self._metrics[mode]["clip_ratio"].append(self.accelerator.gather_for_metrics(clip_ratio).mean().item())
        return loss
