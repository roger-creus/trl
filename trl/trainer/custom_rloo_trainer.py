# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import math
import os
import textwrap
import time
from collections import defaultdict
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.utils import broadcast, gather_object
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    BaseImageProcessor,
    DataCollatorWithPadding,
    FeatureExtractionMixin,
    GenerationConfig,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    Trainer,
    TrainerCallback,
    TrainerControl,
    is_wandb_available,
)
from transformers.integrations import get_reporting_integration_callbacks
from transformers.trainer import DEFAULT_CALLBACKS, DEFAULT_PROGRESS_CALLBACK
from transformers.trainer_callback import CallbackHandler, ExportableState, PrinterCallback
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from ..models.utils import unwrap_model_for_generation
from ..trainer.utils import (
    OnlineTrainerState,
    batch_generation,
    disable_dropout_in_model,
    exact_div,
    first_true_indices,
    forward,
    get_reward,
    prepare_deepspeed,
    print_rich_table,
    selective_log_softmax,
    truncate_response,
)
from .utils import generate_model_card, get_comet_experiment_url, log_table_to_comet_experiment

from IPython import embed

if is_wandb_available():
    import wandb

INVALID_LOGPROB = 1.0


class Custom_RLOOTrainer(Trainer):
    _tag_names = ["trl", "rloo"]

    def __init__(
        self,
        config,  # instance of RLOOConfig
        processing_class: Optional[Union[PreTrainedTokenizerBase, object]] = None,
        policy: nn.Module = None,
        ref_policy: nn.Module = None,
        train_dataset: Dataset = None,
        data_collator=None,
        eval_dataset: Optional[Union[Dataset, dict]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        callbacks: Optional[list] = None,
    ) -> None:
        # Sanity check: ref_policy must be a copy of policy, not the same object.
        if ref_policy is policy:
            raise ValueError(
                "`policy` and `ref_policy` cannot be the same object. If you want `ref_policy` to be the same as `policy`, "
                "make a copy of it, or pass None if using PEFT."
            )
        self.args = config  # RLOOConfig instance
        args = config
        self.processing_class = processing_class
        self.policy = policy

        # If no data collator is provided, create one.
        if data_collator is None:
            from transformers import DataCollatorWithPadding
            data_collator = DataCollatorWithPadding(self.processing_class)
        self.data_collator = data_collator

        # Set generation configuration for the policy.
        self.policy.generation_config.eos_token_id = None
        self.policy.generation_config.pad_token_id = None

        self.ref_policy = ref_policy
        self.train_dataset = train_dataset
        self.train_dataset_len = len(train_dataset)
        self.eval_dataset = eval_dataset
        self.optimizer, self.lr_scheduler = optimizers
        self.optimizer_cls_and_kwargs = None  # for transformers >= 4.47

        # Set total episodes if not provided.
        if args.total_episodes is None:
            args.total_episodes = int(args.num_train_epochs * self.train_dataset_len)

        # Instantiate Accelerator.
        accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
        self.accelerator = accelerator
        if self.accelerator.is_main_process:
            print(f"Using ARGS {config}")
            print("---------------------------------------")
        
        args.world_size = accelerator.num_processes
        args.local_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps * args.num_mini_batches
        args.micro_batch_size = int(args.per_device_train_batch_size * args.world_size)
        args.batch_size = int(args.local_batch_size * args.world_size)
        args.mini_batch_size = exact_div(args.batch_size, args.num_mini_batches, "`batch_size` must be a multiple of `num_mini_batches`")
        args.local_mini_batch_size = exact_div(args.local_batch_size, args.num_mini_batches, "`local_batch_size` must be a multiple of `num_mini_batches`")
        args.num_total_batches = math.ceil(args.total_episodes / args.batch_size)
        time_tensor = torch.tensor(int(time.time()), device=accelerator.device)
        time_int = broadcast(time_tensor, 0).item()
        args.run_name = f"{args.exp_name}__{args.seed}__{time_int}"
        self.local_seed = args.seed + accelerator.process_index * 100003
        if args.num_sample_generations > 0:
            self.sample_generations_freq = max(1, args.num_total_batches // args.num_sample_generations)
        args.local_dataloader_batch_size = exact_div(args.local_batch_size, args.rloo_k, "`local_batch_size` must be a multiple of rloo_k")

        # Set models to evaluation mode (disable dropout)
        for module in [policy, ref_policy]:
            if isinstance(module, nn.Module):
                disable_dropout_in_model(module)
        
        if args.stop_token and args.stop_token == "eos":
            args.stop_token_id = self.processing_class.eos_token_id

        self.model = policy

        # Create optimizer and scheduler (you must implement create_optimizer_and_scheduler)
        self.create_optimizer_and_scheduler(num_training_steps=args.num_total_batches)

        default_callbacks = DEFAULT_CALLBACKS + get_reporting_integration_callbacks(args.report_to)
        self.callbacks = default_callbacks if callbacks is None else default_callbacks + callbacks
        self.callback_handler = CallbackHandler(self.callbacks, self.model, self.processing_class, self.optimizer, self.lr_scheduler)
        self.add_callback(PrinterCallback if args.disable_tqdm else DEFAULT_PROGRESS_CALLBACK)
        
        self.control = TrainerControl()
        self.state = OnlineTrainerState(
            is_local_process_zero=self.is_local_process_zero(),
            is_world_process_zero=self.is_world_process_zero(),
            stateful_callbacks=[cb for cb in self.callback_handler.callbacks + [self.control] if hasattr(cb, "state_dict")]
        )

        self.current_flos = 0
        self.hp_search_backend = None
        self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        self.is_fsdp_enabled = getattr(self.accelerator.state, "fsdp_plugin", None) is not None
        self.hub_model_id = None
        if args.push_to_hub:
            self.init_hf_repo()
        if args.should_save:
            os.makedirs(args.output_dir, exist_ok=True)
        self.backup_model = None

        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

        # ---------------------------
        # Setup dataloaders.
        # ---------------------------
        self.dataloader = DataLoader(
            self.train_dataset,
            batch_size=args.local_dataloader_batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
            drop_last=True,
        )
        torch.manual_seed(args.seed)
        self.model, self.optimizer, self.dataloader = accelerator.prepare(self.model, self.optimizer, self.dataloader)
        torch.manual_seed(self.local_seed)

        self.eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=args.per_device_eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=True,
        )
        self.eval_dataloader = accelerator.prepare(self.eval_dataloader)

        if self.is_deepspeed_enabled:
            self.ref_policy = prepare_deepspeed(self.ref_policy, args.per_device_train_batch_size, args.fp16, args.bf16)
            self.deepspeed = self.model
        else:
            self.ref_policy = self.ref_policy.to(self.accelerator.device)
            
    def get_train_dataloader(self):
        return self.dataloader

    def get_eval_dataloader(self):
        return self.eval_dataloader

    def train(self):
        args = self.args
        accelerator = self.accelerator
        optimizer = self.optimizer
        model = self.model
        self.model_wrapped = self.model
        ref_policy = self.ref_policy
        processing_class = self.processing_class
        dataloader = self.dataloader
        device = accelerator.device

        def repeat_generator():
            while True:
                yield from dataloader

        iter_dataloader = iter(repeat_generator())
        from transformers import GenerationConfig
        generation_config = GenerationConfig(
            max_new_tokens=args.response_length,
            temperature=(args.temperature + 1e-7),
            top_k=0.0,
            top_p=1.0,
            do_sample=True,
        )

        accelerator.print("=== Training policy ===")
        start_time = time.time()
        stats_shape = (args.num_ppo_epochs, args.num_mini_batches, args.gradient_accumulation_steps)
        approxkl_stats = torch.zeros(stats_shape, device=device)
        pg_clipfrac_stats = torch.zeros(stats_shape, device=device)
        pg_loss_stats = torch.zeros(stats_shape, device=device)
        vf_clipfrac_stats = torch.zeros(stats_shape, device=device)
        entropy_stats = torch.zeros(stats_shape, device=device)
        ratio_stats = torch.zeros(stats_shape, device=device)
        model.train()

        self.state.global_step = 0
        self.state.episode = 0
        self.state.max_steps = (args.num_total_batches * args.num_mini_batches) // 2
        self.state.num_train_epochs = args.total_episodes / self.train_dataset_len
        if args.logging_steps is not None:
            self.state.logging_steps = args.logging_steps if args.logging_steps >= 1 else math.ceil(self.state.max_steps * args.logging_steps)
        if args.eval_steps is not None:
            self.state.eval_steps = args.eval_steps if args.eval_steps >= 1 else math.ceil(self.state.max_steps * args.eval_steps)
        if args.save_steps is not None:
            self.state.save_steps = args.save_steps if args.save_steps >= 1 else math.ceil(self.state.max_steps * args.save_steps)
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        for update in range(1, args.num_total_batches + 1):
            self.state.episode += args.batch_size
            data = next(iter_dataloader)
            with torch.no_grad():
                queries = data["input_ids"].to(device)
                queries = queries.repeat(args.rloo_k, 1)
                context_length = queries.shape[1]
                responses = []
                postprocessed_responses = []
                logprobs = []
                ref_logprobs = []
                scores = []
                sequence_lengths = []
                
                # Generate responses and compute logprobs
                with unwrap_model_for_generation(
                    self.model, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
                ) as unwrapped_model:
                    query_responses, logitss = batch_generation(
                        unwrapped_model,
                        queries,
                        args.local_rollout_forward_batch_size,
                        processing_class.pad_token_id,
                        generation_config,
                    )

                for i in range(0, queries.shape[0], args.local_rollout_forward_batch_size):
                    query = queries[i : i + args.local_rollout_forward_batch_size]
                    query_response = query_responses[i : i + args.local_rollout_forward_batch_size]
                    response = query_response[:, context_length:]
                    logits = logitss[i : i + args.local_rollout_forward_batch_size]
                    logprob = selective_log_softmax(logits, response)
                    del logits
                    torch.cuda.empty_cache()

                    ref_output = forward(ref_policy, query_response, processing_class.pad_token_id)
                    ref_logits = ref_output.logits[:, context_length - 1 : -1]
                    ref_logits /= args.temperature + 1e-7
                    ref_logprob = selective_log_softmax(ref_logits, response)
                    del ref_output, ref_logits
                    torch.cuda.empty_cache()

                    # Response Processing 1. truncate response after the first occurrence of `stop_token_id`
                    postprocessed_response = response
                    if args.stop_token_id is not None:  # handle the edge case when stop_token_id exists but is 0
                        postprocessed_response = truncate_response(
                            args.stop_token_id, processing_class.pad_token_id, response
                        )

                    sequence_length = first_true_indices(postprocessed_response == processing_class.pad_token_id) - 1

                    # For evaluation, decode query and response separately.
                    decoded_queries = processing_class.batch_decode(query, skip_special_tokens=True)
                    decoded_completions = processing_class.batch_decode(postprocessed_response, skip_special_tokens=True)
                    eval_prompts = []
                    for q, c in zip(decoded_queries, decoded_completions):
                        actual_q = q.split("### Your turn ###")[1].strip()
                        try:
                            reasoning_chain = c.split("</think>")[0].strip()
                            final_part = c.split("</think>")[1]
                            additional_reasoning = final_part.split("ANSWER:")[0].strip()
                            reasoning_chain += f"\n\n{additional_reasoning}"
                            predicted_answer = final_part.split("ANSWER:")[1].strip()
                        except:
                            reasoning_chain = c
                            predicted_answer = "-1"
                        
                        prompt_text = (
                            "Your task is to determine whether the provided GUESS for a multiple-choice question is CORRECT or INCORRECT. "
                            "You do not need to know the true answer; instead, analyse the accompanying REASONING and evaluate the GUESS.\n\n"
                            
                            "Input Details:\n"
                            "1. QUESTION: A multiple-choice question with several answer options.\n"
                            "2. REASONING: A step-by-step explanation supporting the GUESS.\n"
                            "3. GUESS: The selected answer in the format 'GUESS: <integer_idx>'.\n\n"
                            
                            "Special Instructions:\n"
                            "- If the GUESS is '-1', immediately return INCORRECT without further analysis.\n"
                            "- Your final response must end with exactly one of the following lines (with no additional text):\n"
                            "    EVALUATION: INCORRECT\n"
                            "    EVALUATION: CORRECT\n\n"
                            
                            "Evaluation Guidelines:\n"
                            "- Although your final verdict must depend only on the GUESS, carefully analyze the provided REASONING for any logical flaws.\n"
                            "- Think deeply and be thorough in your analysis.\n"
                            f"- You have a maximum of {args.response_length // 2} words to generate your response, so be clear and concise.\n\n"
                            
                            "### QUESTION ###\n"
                            f"{actual_q}\n\n"
                            
                            "### REASONING ###\n"
                            f"{reasoning_chain}\n\n"
                            
                            "### GUESS ###\n"
                            f"{predicted_answer}\n\n"
                            
                            "### Your turn ###\n"
                        )
                        eval_prompts.append(prompt_text)

                    # Run batched evaluation using the evaluator model with response truncation.
                    with unwrap_model_for_generation(
                        self.ref_policy,
                        self.accelerator,
                        gather_deepspeed3_params=args.ds3_gather_for_generation
                    ) as unwrapped_evaluator:
                        
                        # Prepare the inputs for the evaluator.
                        eval_inputs = self.processing_class(eval_prompts, return_tensors="pt", padding=True)
                        eval_inputs = {k: v.to(device) for k, v in eval_inputs.items()}
                        
                        with torch.no_grad():
                            # Get the input_ids from the prepared prompt.
                            eval_query = eval_inputs["input_ids"]
                            eval_context_length = eval_query.shape[1]
                            
                            # Generate responses in batch using the batch_generation function.
                            eval_responses, _ = batch_generation(
                                unwrapped_evaluator,
                                eval_query,
                                eval_query.shape[0],
                                self.processing_class.pad_token_id,
                                generation_config,
                            )
                            
                            # Extract tokens generated beyond the prompt.
                            eval_responses = eval_responses[:, eval_context_length:]
                            
                            if args.stop_token_id is not None:
                                postprocessed_eval_response = truncate_response(
                                    args.stop_token_id,
                                    self.processing_class.pad_token_id,
                                    eval_responses
                                )
                            else:
                                postprocessed_eval_response = eval_responses
                        
                        # Decode the postprocessed responses into text.
                        eval_texts = self.processing_class.batch_decode(postprocessed_eval_response, skip_special_tokens=True)

                    # Parse outputs to obtain binary rewards.
                    batch_rewards = []
                    for txt in eval_texts:
                        try:
                            evaluation_txt = txt.strip().split("\n")
                            evaluation_txt = next((line for line in evaluation_txt if "EVALUATION:" in line), None)
                            reward_value = int(1 if evaluation_txt.replace("EVALUATION:", "").strip() == "CORRECT" else 0)
                            if reward_value not in [0, 1]:
                                raise ValueError("Invalid evaluation score")
                        except Exception as e:
                            reward_value = 0
                            print(f"Error parsing evaluator output: {e}")
                        
                        batch_rewards.append(reward_value)

                    score = torch.tensor(batch_rewards, dtype=torch.float, device=device)
                    responses.append(response)
                    postprocessed_responses.append(postprocessed_response)
                    logprobs.append(logprob)
                    ref_logprobs.append(ref_logprob)
                    sequence_lengths.append(sequence_length)
                    scores.append(score)

                responses = torch.cat(responses, 0)
                postprocessed_responses = torch.cat(postprocessed_responses, 0)
                logprobs = torch.cat(logprobs, 0)
                ref_logprobs = torch.cat(ref_logprobs, 0)
                sequence_lengths = torch.cat(sequence_lengths, 0)
                scores = torch.cat(scores, 0)
                del logprob, score, ref_logprob, eval_prompts, eval_inputs, eval_responses, postprocessed_eval_response, eval_texts, eval_query
                torch.cuda.empty_cache()
                gc.collect()

                # If responses do not contain the stop token, apply a penalty.
                contain_eos_token = torch.any(postprocessed_responses == processing_class.eos_token_id, dim=-1)
                if args.missing_eos_penalty is not None:
                    scores[~contain_eos_token] -= args.missing_eos_penalty

                print(f"Scores: {scores}")

                response_idxs = torch.arange(responses.shape[1], device=responses.device).repeat(responses.shape[0], 1)
                padding_mask = response_idxs > sequence_lengths.unsqueeze(1)
                logprobs = torch.masked_fill(logprobs, padding_mask, INVALID_LOGPROB)
                ref_logprobs = torch.masked_fill(ref_logprobs, padding_mask, INVALID_LOGPROB)

                kl = logprobs - ref_logprobs

                if args.normalize_reward:
                    scores = (scores - scores.mean()) / (scores.std() + 1e-8)
                    scores = torch.clamp(scores, -args.reward_clip_range, args.reward_clip_range)

                if args.token_level_kl:
                    kl_reward = -args.kl_coef * kl
                    eos_indices = padding_mask.size(1) - 1 - padding_mask.long().fliplr().argmax(dim=1, keepdim=True)
                    last_reward = torch.zeros_like(kl)
                    scores_shaped = scores.reshape(-1, 1).to(kl.dtype)
                    last_reward.scatter_(dim=1, index=eos_indices, src=scores_shaped)
                    non_score_reward = kl_reward.sum(1)
                    reward = last_reward + kl_reward
                    rlhf_reward = reward.sum(1)
                else:
                    sequence_kl = kl.sum(1)
                    non_score_reward = -args.kl_coef * sequence_kl
                    rlhf_reward = non_score_reward + scores

                rlhf_reward = rlhf_reward.reshape(args.rloo_k, -1)
                baseline = (rlhf_reward.sum(0) - rlhf_reward) / (args.rloo_k - 1)
                advantages = rlhf_reward - baseline
                advantages = advantages.flatten()

                if args.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                torch.cuda.empty_cache()

            # Do multiple epochs of PPO training, with a fresh random shuffle in each epoch
            for ppo_epoch_idx in range(args.num_ppo_epochs):
                b_inds = np.random.permutation(args.local_batch_size)
                minibatch_idx = 0
                for mini_batch_start in range(0, args.local_batch_size, args.local_mini_batch_size):
                    mini_batch_end = mini_batch_start + args.local_mini_batch_size
                    mini_batch_inds = b_inds[mini_batch_start:mini_batch_end]
                    gradient_accumulation_idx = 0
                    for micro_batch_start in range(0, args.local_mini_batch_size, args.per_device_train_batch_size):
                        with accelerator.accumulate(model):
                            micro_batch_end = micro_batch_start + args.per_device_train_batch_size
                            micro_batch_inds = mini_batch_inds[micro_batch_start:micro_batch_end]

                            # Get batch data
                            mb_advantage = advantages[micro_batch_inds]
                            mb_responses = responses[micro_batch_inds]
                            mb_query_responses = query_responses[micro_batch_inds]
                            mb_logprobs = logprobs[micro_batch_inds]

                            # Forward pass
                            output = forward(model, mb_query_responses, processing_class.pad_token_id)
                            logits = output.logits[:, context_length - 1 : -1]
                            logits /= args.temperature + 1e-7

                            # Compute new logprobs
                            new_logprobs = selective_log_softmax(logits, mb_responses)
                            new_logprobs = torch.masked_fill(
                                new_logprobs, padding_mask[micro_batch_inds], INVALID_LOGPROB
                            )

                            # Compute probability ratios
                            new_ratio = (new_logprobs - mb_logprobs).exp()
                            new_logprobs = new_logprobs.sum(1)
                            mb_logprobs = mb_logprobs.sum(1)
                            logprobs_diff = new_logprobs - mb_logprobs
                            ratio = torch.exp(logprobs_diff)

                            # PPO clipped loss
                            pg_losses = -mb_advantage * ratio
                            pg_losses2 = -mb_advantage * torch.clamp(ratio, 1.0 - args.cliprange, 1.0 + args.cliprange)
                            pg_loss_max = torch.max(pg_losses, pg_losses2)
                            pg_loss = pg_loss_max.mean()

                            # Final loss
                            loss = pg_loss

                            # Optimization step
                            accelerator.backward(loss)
                            optimizer.step()
                            optimizer.zero_grad()

                            with torch.no_grad():
                                pg_clipfrac = (pg_losses2 > pg_losses).float().mean()
                                prob_dist = torch.nn.functional.softmax(logits, dim=-1)
                                entropy = torch.logsumexp(logits, dim=-1) - torch.sum(prob_dist * logits, dim=-1)
                                approxkl = 0.5 * (logprobs_diff**2).mean()
                                approxkl_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = approxkl
                                pg_clipfrac_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = (
                                    pg_clipfrac
                                )
                                pg_loss_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = pg_loss
                                entropy_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = entropy.mean()
                                ratio_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = new_ratio.mean()
                        gradient_accumulation_idx += 1
                    minibatch_idx += 1

                    # del everything and empty cache
                    # fmt: off
                    del (
                        output, logits, new_logprobs, logprobs_diff, ratio, pg_losses,
                        pg_losses2, pg_loss, loss, pg_clipfrac, prob_dist, entropy, approxkl,
                        mb_advantage, mb_responses, mb_query_responses, mb_logprobs,
                    )
                    # fmt: on
                    torch.cuda.empty_cache()

            # Compute metrics
            with torch.no_grad():
                mean_kl = kl.sum(1).mean()
                mean_entropy = (-logprobs).sum(1).mean()
                mean_non_score_reward = non_score_reward.mean()
                eps = int(self.state.episode / (time.time() - start_time))
                metrics = {}
                metrics["eps"] = eps
                metrics["objective/kl"] = self.accelerator.gather_for_metrics(mean_kl).mean().item()
                metrics["objective/entropy"] = self.accelerator.gather_for_metrics(mean_entropy).mean().item()
                metrics["objective/non_score_reward"] = (
                    self.accelerator.gather_for_metrics(mean_non_score_reward).mean().item()
                )
                metrics["objective/rlhf_reward"] = self.accelerator.gather_for_metrics(rlhf_reward).mean().item()
                metrics["objective/scores"] = self.accelerator.gather_for_metrics(scores.mean()).mean().item()
                metrics["policy/approxkl_avg"] = self.accelerator.gather_for_metrics(approxkl_stats).mean().item()
                metrics["policy/clipfrac_avg"] = self.accelerator.gather_for_metrics(pg_clipfrac_stats).mean().item()
                metrics["loss/policy_avg"] = self.accelerator.gather_for_metrics(pg_loss_stats).mean().item()
                metrics["val/clipfrac_avg"] = self.accelerator.gather_for_metrics(vf_clipfrac_stats).mean().item()
                metrics["policy/entropy_avg"] = self.accelerator.gather_for_metrics(entropy_stats).mean().item()
                metrics["val/ratio"] = self.accelerator.gather_for_metrics(ratio_stats).mean().item()
                metrics["val/ratio_var"] = self.accelerator.gather_for_metrics(ratio_stats).var().item()
                metrics["val/num_eos_tokens"] = (responses == processing_class.eos_token_id).sum().item()
                metrics["lr"] = self.lr_scheduler.get_last_lr()[0]
                metrics["episode"] = self.state.episode
                self.state.epoch = self.state.episode / (args.rloo_k * self.train_dataset_len)  # used by self.log
                self.log(metrics)
            del kl, mean_kl, mean_entropy, scores

            self.lr_scheduler.step()
            self.state.global_step += 1
            self.control = self.callback_handler.on_step_end(args, self.state, self.control)
            if self.control.should_save:
                self._save_checkpoint(model, trial=None)
                self.control = self.callback_handler.on_save(self.args, self.state, self.control)
            torch.cuda.empty_cache()
            gc.collect()

            if args.num_sample_generations > 0 and (update - 1) % self.sample_generations_freq == 0:
                self.generate_completions(sampling=True)

        # HF trainer specifics
        self.control = self.callback_handler.on_train_end(args, self.state, self.control)
        if self.control.should_save:
            self._save_checkpoint(model, trial=None, metrics=None)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def generate_completions(self, sampling: bool = False):
        args = self.args
        processing_class = self.processing_class
        device = self.accelerator.device
        from transformers import GenerationConfig
        generation_config = GenerationConfig(
            max_new_tokens=args.response_length,
            temperature=0.01 + 1e-7,
            top_k=0.0,
            top_p=1.0,
            do_sample=True,
        )
        table = defaultdict(list)
        
        # Use the model for generation.
        with unwrap_model_for_generation(self.model, self.accelerator, gather_deepspeed3_params=args.ds3_gather_for_generation) as unwrapped_model:
            for batch in self.eval_dataloader:
                query = batch["input_ids"]
                with torch.no_grad():
                    context_length = query.shape[1]
                    # Generate responses in batch.
                    query_responses, _ = batch_generation(
                        unwrapped_model,
                        query,
                        query.shape[0],
                        processing_class.pad_token_id,
                        generation_config,
                    )
                    # Extract generated tokens beyond the prompt.
                    response = query_responses[:, context_length:]
                    postprocessed_response = response
                    if args.stop_token_id is not None:
                        postprocessed_response = truncate_response(
                            args.stop_token_id, processing_class.pad_token_id, response
                        )
                    
                    # Save original query and generated response (for display).
                    table["query"].extend(gather_object(processing_class.batch_decode(query, skip_special_tokens=True)))
                    table["model response"].extend(gather_object(processing_class.batch_decode(postprocessed_response)))
                    
                    # For evaluation, decode query and response separately.
                    decoded_queries = processing_class.batch_decode(query, skip_special_tokens=True)
                    decoded_completions = processing_class.batch_decode(postprocessed_response, skip_special_tokens=True)
                    eval_prompts = []
                    for q, c in zip(decoded_queries, decoded_completions):
                        actual_q = q.split("### Your turn ###")[1].strip()
                        try:
                            reasoning_chain = c.split("</think>")[0].strip()
                            final_part = c.split("</think>")[1]
                            additional_reasoning = final_part.split("ANSWER:")[0].strip()
                            reasoning_chain += f"\n\n{additional_reasoning}"
                            predicted_answer = final_part.split("ANSWER:")[1].strip()
                        except:
                            reasoning_chain = c
                            predicted_answer = "-1"
                        
                        prompt_text = (
                            "Your task is to determine whether the provided GUESS for a multiple-choice question is CORRECT or INCORRECT. "
                            "You do not need to know the true answer; instead, analyse the accompanying REASONING and evaluate the GUESS.\n\n"
                            
                            "Input Details:\n"
                            "1. QUESTION: A multiple-choice question with several answer options.\n"
                            "2. REASONING: A step-by-step explanation supporting the GUESS.\n"
                            "3. GUESS: The selected answer in the format 'GUESS: <integer_idx>'.\n\n"
                            
                            "Special Instructions:\n"
                            "- If the GUESS is '-1', immediately return INCORRECT without further analysis.\n"
                            "- Your final response must end with exactly one of the following lines (with no additional text):\n"
                            "    EVALUATION: INCORRECT\n"
                            "    EVALUATION: CORRECT\n\n"
                            
                            "Evaluation Guidelines:\n"
                            "- Although your final verdict must depend only on the GUESS, carefully analyze the provided REASONING for any logical flaws.\n"
                            "- Think deeply and be thorough in your analysis.\n"
                            f"- You have a maximum of {args.response_length // 2} words to generate your response, so be clear and concise.\n\n"
                            
                            "### QUESTION ###\n"
                            f"{actual_q}\n\n"
                            
                            "### REASONING ###\n"
                            f"{reasoning_chain}\n\n"
                            
                            "### GUESS ###\n"
                            f"{predicted_answer}\n\n"
                            
                            "### Your turn ###\n"
                        )
                        eval_prompts.append(prompt_text)

                    # Run batched evaluation using the evaluator model with response truncation.
                    with unwrap_model_for_generation(
                        self.ref_policy,
                        self.accelerator,
                        gather_deepspeed3_params=args.ds3_gather_for_generation
                    ) as unwrapped_evaluator:
                        
                        # Prepare the inputs for the evaluator.
                        eval_inputs = self.processing_class(eval_prompts, return_tensors="pt", padding=True)
                        eval_inputs = {k: v.to(device) for k, v in eval_inputs.items()}
                        
                        # Get the input_ids from the prepared prompt.
                        eval_query = eval_inputs["input_ids"]
                        eval_context_length = eval_query.shape[1]
                        
                        # Generate responses in batch using the batch_generation function.
                        eval_responses, _ = batch_generation(
                            unwrapped_evaluator,
                            eval_query,
                            eval_query.shape[0],
                            self.processing_class.pad_token_id,
                            generation_config,
                        )
                        
                        # Extract tokens generated beyond the prompt.
                        eval_response = eval_responses[:, eval_context_length:]
                        
                        if args.stop_token_id is not None:
                            eval_postprocessed_response = truncate_response(
                                args.stop_token_id,
                                self.processing_class.pad_token_id,
                                eval_response
                            )
                        else:
                            eval_postprocessed_response = eval_response
                    
                    # Decode the postprocessed responses into text.
                    eval_texts = self.processing_class.batch_decode(eval_postprocessed_response, skip_special_tokens=True)
                    
                    # Parse outputs to obtain binary rewards.
                    batch_rewards = []
                    for txt in eval_texts:
                        try:
                            evaluation_txt = txt.strip().split("\n")
                            evaluation_txt = next((line for line in evaluation_txt if "EVALUATION:" in line), None)
                            reward_value = int(1 if evaluation_txt.replace("EVALUATION:", "").strip() == "CORRECT" else 0)
                            if reward_value not in [0, 1]:
                                raise ValueError("Invalid evaluation score")
                        except Exception as e:
                            reward_value = 0
                            print(f"Error parsing evaluator output: {e}")
                        
                        batch_rewards.append(reward_value)

                    score = torch.tensor(batch_rewards, dtype=torch.float, device=device)

                    if args.missing_eos_penalty is not None:
                        contain_eos_token = torch.any(postprocessed_response == processing_class.eos_token_id, dim=-1)
                        score[~contain_eos_token] -= args.missing_eos_penalty
                    
                    table["score"].extend(self.accelerator.gather_for_metrics(score).float().cpu().numpy())
                    table["evaluator response"].extend(gather_object(eval_texts))
                if sampling:
                    break
        
        import pandas as pd
        df = pd.DataFrame(table)
        if self.accelerator.is_main_process:
            print_rich_table(df.iloc[:5])
            if "wandb" in args.report_to:
                import wandb
                if wandb.run is not None:
                    wandb.log({"completions": wandb.Table(dataframe=df)}, step=self.state.global_step)
            if "comet_ml" in args.report_to:
                log_table_to_comet_experiment(name="completions.csv", table=df)
                
        del query, query_responses, response, postprocessed_response, decoded_queries, decoded_completions, eval_prompts, eval_inputs, eval_responses, eval_response, eval_postprocessed_response, eval_texts, batch_rewards, score, table, df
        torch.cuda.empty_cache()
        gc.collect()

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent("""\
        @inproceedings{ahmadian2024back,
            title        = {{Back to Basics: Revisiting REINFORCE-Style Optimization for Learning from Human Feedback in LLMs}},
            author       = {Arash Ahmadian and Chris Cremer and Matthias Gall{\'{e}} and Marzieh Fadaee and Julia Kreutzer and Olivier Pietquin and Ahmet {\"{U}}st{\"{u}}n and Sara Hooker},
            year         = 2024,
            booktitle    = {Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), {ACL} 2024, Bangkok, Thailand, August 11-16, 2024},
            publisher    = {Association for Computational Linguistics},
            pages        = {12248--12267},
            editor       = {Lun{-}Wei Ku and Andre Martins and Vivek Srikumar},
        }""")

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="RLOO",
            trainer_citation=citation,
            paper_title="Back to Basics: Revisiting REINFORCE-Style Optimization for Learning from Human Feedback in LLMs",
            paper_id="2402.14740",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))
