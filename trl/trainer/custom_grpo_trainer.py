import os
import textwrap
import warnings
from collections import defaultdict
from typing import Any, Callable, Optional, Sized, Union
from unittest.mock import patch

import torch
import torch.utils.data
import transformers
from accelerate.utils import broadcast_object_list, gather, gather_object, set_seed
from accelerate.utils.other import is_compiled_module
from datasets import Dataset, IterableDataset
from packaging import version
from torch import nn
from torch.utils.data import Sampler
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from ..data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from ..import_utils import is_vllm_available
from ..models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from .callbacks import SyncRefModelCallback
from .grpo_config import GRPOConfig
from .utils import generate_model_card, get_comet_experiment_url, pad, selective_log_softmax

from IPython import embed

if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_vllm_available():
    from vllm import LLM, SamplingParams

if is_wandb_available():
    import wandb

RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]

class RepeatRandomSampler(Sampler):
    """
    Sampler that repeats the indices of a dataset N times.

    Args:
        data_source (`Sized`):
            Dataset to sample from.
        repeat_count (`int`):
            Number of times to repeat each index.
        seed (`Optional[int]`):
            Random seed for reproducibility (only affects this sampler).

    Example:
    ```python
    >>> sampler = RepeatRandomSampler(["a", "b", "c", "d"], repeat_count=2)
    >>> list(sampler)
    [2, 2, 0, 0, 3, 3, 1, 1]
    ```
    """

    def __init__(self, data_source: Sized, repeat_count: int, seed: Optional[int] = None):
        self.data_source = data_source
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.seed = seed
        self.generator = torch.Generator()  # Create a local random generator
        if seed is not None:
            self.generator.manual_seed(seed)

    def __iter__(self):
        indexes = [
            idx
            for idx in torch.randperm(self.num_samples, generator=self.generator).tolist()
            for _ in range(self.repeat_count)
        ]
        return iter(indexes)

    def __len__(self):
        return self.num_samples * self.repeat_count

class Custom_GRPOTrainer(Trainer):
    _tag_names = ["trl", "grpo"]

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list] = None,
        optimizers: tuple[
            Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]
        ] = (None, None),
        peft_config: Optional[Any] = None,
        # NEW: Parameters for evaluator LLM for reward computation:
        evaluator_tokenizer: Optional[PreTrainedTokenizerBase] = None,
        evaluator_sampling_params: Optional[Any] = None,
    ):
        # Initialize args if not provided.
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        # ---------------------------
        # Model Initialization Block
        # ---------------------------
        model_init_kwargs = args.model_init_kwargs or {}
        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass
            elif isinstance(torch_dtype, str):
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError("Invalid `torch_dtype` passed to `GRPOConfig`.")
            model_init_kwargs["use_cache"] = False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError("You passed `model_init_kwargs` to `GRPOConfig`, but your model is already instantiated.")

        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        # Create reference model
        if is_deepspeed_zero3_enabled():
            self.ref_model = AutoModelForCausalLM.from_pretrained(model_id, **model_init_kwargs)
        elif peft_config is None:
            self.ref_model = create_reference_model(model)
        else:
            self.ref_model = None

        # -------------------------------
        # Processing and Reward Functions
        # -------------------------------
        if processing_class is None:
            processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")

        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        if args.reward_weights is not None:
            if len(args.reward_weights) != len(reward_funcs):
                raise ValueError(
                    f"Number of reward weights ({len(args.reward_weights)}) must match number of reward functions ({len(reward_funcs)})"
                )
            self.reward_weights = torch.tensor(args.reward_weights, dtype=torch.float32)
        else:
            self.reward_weights = torch.ones(len(reward_funcs), dtype=torch.float32)

        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        def data_collator(features):
            return features

        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length  # number of tokens in completion
        self.num_generations = args.num_generations  # number of completions per prompt
        self.use_vllm = args.use_vllm
        self.beta = args.beta

        model.warnings_issued["estimate_tokens"] = True

        self._metrics = defaultdict(list)
        self.log_completions = args.log_completions

        # -----------------------------
        # NEW: Save evaluator parameters
        # -----------------------------
        self.evaluator_tokenizer = evaluator_tokenizer
        self.evaluator_sampling_params = evaluator_sampling_params

        # Call parent constructor.
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # Additional configuration checks.
        num_processes = self.accelerator.num_processes
        global_batch_size = args.per_device_train_batch_size * num_processes
        possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if global_batch_size % n_gen == 0]
        if self.num_generations not in possible_values:
            raise ValueError(
                f"The global train batch size must be evenly divisible by the number of generations per prompt. Valid values: {possible_values}."
            )
        if self.args.eval_strategy != "no":
            global_batch_size = args.per_device_eval_batch_size * num_processes
            possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if global_batch_size % n_gen == 0]
            if self.num_generations not in possible_values:
                raise ValueError(
                    f"The global eval batch size must be evenly divisible by the number of generations per prompt. Valid values: {possible_values}."
                )

        set_seed(args.seed, device_specific=True)

        if self.use_vllm:
            if self.accelerator.is_main_process:
                vllm_device = self.args.vllm_device
                if vllm_device == "auto":
                    if torch.cuda.device_count() == 1:
                        vllm_device = "cuda:0"
                    else:
                        vllm_device = f"cuda:{self.accelerator.num_processes}"
                if vllm_device.split(":")[0] == "cuda" and int(vllm_device.split(":")[1]) >= torch.cuda.device_count():
                    raise ValueError(f"The requested device for vllm ({vllm_device}) is not available.")
                if vllm_device in {f"cuda:{idx}" for idx in range(self.accelerator.num_processes)}:
                    warnings.warn(
                        f"The requested device {vllm_device} is also used for training. For higher throughput, consider using a dedicated device for vLLM."
                    )
                from unittest.mock import patch
                world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
                profiling_patch = patch("vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling", return_value=None)
                with world_size_patch, profiling_patch:
                    self.llm = LLM(
                        model=model.name_or_path,
                        device=vllm_device,
                        gpu_memory_utilization=self.args.vllm_gpu_memory_utilization,
                        dtype=self.args.vllm_dtype,
                        enable_prefix_caching=True,
                        max_model_len=self.args.vllm_max_model_len,
                    )
                self.sampling_params = SamplingParams(
                    temperature=args.temperature,
                    max_tokens=self.max_completion_length,
                )
            self._last_loaded_step = 0
            self.accelerator.wait_for_everyone()
        else:
            self.generation_config = GenerationConfig(
                max_new_tokens=self.max_completion_length,
                do_sample=True,
                temperature=args.temperature,
                pad_token_id=processing_class.pad_token_id,
            )
        self.model_accepts_loss_kwargs = False
        self.model.add_model_tags(self._tag_names)

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
        if args.sync_ref_model:
            self.add_callback(SyncRefModelCallback(ref_model=self.ref_model, accelerator=self.accelerator))
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    def _get_train_sampler(self) -> torch.utils.data.Sampler:
        return RepeatRandomSampler(self.train_dataset, self.num_generations, seed=self.args.seed)

    def _get_eval_sampler(self, eval_dataset) -> torch.utils.data.Sampler:
        return RepeatRandomSampler(eval_dataset, self.num_generations, seed=self.args.seed)

    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep):
        logits = model(
            input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=logits_to_keep + 1
        ).logits
        logits = logits[:, :-1, :]
        input_ids = input_ids[:, -logits_to_keep:]
        logits = logits[:, -logits_to_keep:]
        return selective_log_softmax(logits, input_ids)

    def _move_model_to_vllm(self):
        with unwrap_model_for_generation(
            self.model, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
        ) as unwrapped_model:
            if is_compiled_module(unwrapped_model):
                state_dict = unwrapped_model._orig_mod.state_dict()
            else:
                state_dict = unwrapped_model.state_dict()
        if self.accelerator.is_main_process:
            llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
            llm_model.load_weights(state_dict.items())

    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        prompts = [x["prompts"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompts"] for example in inputs]
        prompt_inputs = self.processing_class(
            prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length:]
            prompt_mask = prompt_mask[:, -self.max_prompt_length:]
        if self.args.use_vllm:
            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step
            all_prompts_text = gather_object(prompts_text)
            if self.accelerator.is_main_process:
                outputs = self.llm.generate(all_prompts_text, sampling_params=self.sampling_params, use_tqdm=False)
                completion_ids = [out.token_ids for completions in outputs for out in completions.outputs]
            else:
                completion_ids = [None] * len(all_prompts_text)
            completion_ids = broadcast_object_list(completion_ids, from_process=0)
            process_slice = slice(
                self.accelerator.process_index * len(prompts),
                (self.accelerator.process_index + 1) * len(prompts),
            )
            completion_ids = completion_ids[process_slice]
            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
            completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        else:
            with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
                prompt_completion_ids = unwrapped_model.generate(
                    prompt_ids, attention_mask=prompt_mask, generation_config=self.generation_config
                )
            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        with torch.inference_mode():
            if self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model, prompt_completion_ids, attention_mask, logits_to_keep
                    )
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        # ---------------
        # Reward Computation
        # ---------------
        # Build evaluation prompts.
        eval_prompts = []
        for prompt, completion in zip(prompts, completions):
            eval_prompt = (
                "You are an evaluator that checks if the provided reasoning and response for a multiple-choice question is correct. "
                "Given the question and the reasoning with the answer, your final output needs to be only a 1 if you believe the reasoning is correct and the answer is right; output 0 otherwise. "
                "Importantly, your last line should only be 'Answer: <integer_idx>' with no additional contents.\n"
                f"Question: {prompt}\n\n"
                f"Response: {completion}\n\n"
            )
            eval_input = [{"role": "user", "content": eval_prompt}]
            formatted_eval_prompt = self.evaluator_tokenizer.apply_chat_template(
                eval_input, tokenize=False, add_generation_prompt=True
            )
            eval_prompts.append(formatted_eval_prompt)
        
        # Use batched inference for evaluation.
        with torch.inference_mode():
            eval_inputs = self.evaluator_tokenizer(eval_prompts, return_tensors="pt", padding=True)
            eval_inputs = {k: v.to(device) for k, v in eval_inputs.items()}
            output_ids = self.ref_model.generate(
                **eval_inputs,
                max_new_tokens=self.evaluator_sampling_params.max_tokens,
                temperature=self.evaluator_sampling_params.temperature
            )
            eval_texts = self.evaluator_tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        
        # Mimic vLLM output structure for uniform downstream processing.
        class DummyOutput:
            def __init__(self, text):
                self.text = text
        class DummyResponse:
            def __init__(self, text):
                self.outputs = [DummyOutput(text)]
        eval_responses = [DummyResponse(text) for text in eval_texts]
        evaluator_rewards = []
        for response in eval_responses:
            raw_eval = response.outputs[0].text.strip() if response.outputs and response.outputs[0].text else ""
            try:
                raw_eval = raw_eval.split("<｜Assistant｜>")[1].strip()
                raw_eval = raw_eval.split("</think>")[1].strip()
                raw_eval = raw_eval.split("Answer:")[1].strip()
                binary_signal = int(raw_eval)
                if binary_signal not in [0, 1]:
                    binary_signal = 0
            except Exception as e:
                print(f"Error: {e}")
                binary_signal = 0
            evaluator_rewards.append(binary_signal)
        
        print(evaluator_rewards)

        rewards_per_func = torch.tensor(evaluator_rewards, dtype=torch.float32, device=device).unsqueeze(1)
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).sum(dim=1)
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]
        reward_per_func = rewards_per_func.mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, nn.Module):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())
        self._metrics["reward"].append(rewards.mean().item())
        self._metrics["reward_std"].append(std_grouped_rewards.mean().item())
        if (
            self.log_completions
            and self.state.global_step % self.args.logging_steps == 0
            and "wandb" in self.args.report_to
        ):
            import pandas as pd
            table = {
                "step": [str(self.state.global_step)] * len(rewards),
                "prompt": gather_object(prompts_text),
                "completion": gather_object(completions_text),
                "reward": rewards.tolist(),
            }
            df = pd.DataFrame(table)
            if hasattr(__import__("wandb"), "run") and self.accelerator.is_main_process:
                import wandb
                wandb.log({"completions": wandb.Table(dataframe=df)})
                
        # clean from gpu device (e.g. eval_inputs..)
        del eval_inputs
        
        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
        }

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)
        ref_per_token_logps = inputs["ref_per_token_logps"]
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
        advantages = inputs["advantages"]
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)
        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())
        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys: Optional[list[str]] = None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            loss = loss.mean().detach()
        return loss, None, None

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        from packaging import version
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}
        if next(iter(logs.keys())).startswith("eval_"):
            metrics = {f"eval_{key}": val for key, val in metrics.items()}
        logs = {**logs, **metrics}
        import transformers
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:
            super().log(logs)
        self._metrics.clear()

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
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

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            }
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))
