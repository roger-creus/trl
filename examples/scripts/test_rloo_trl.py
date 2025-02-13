import shutil
from accelerate import PartialState
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
)
from trl import ModelConfig, RLOOConfig, RLOOTrainer, ScriptArguments
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE
from trl import Custom_RLOOTrainer as RLOOTrainer

"""
python examples/scripts/test_rloo_trl.py \
    --dataset_name cais/mmlu \
    --dataset_train_split test \
    --learning_rate 3e-6 \
    --num_ppo_epochs 1 \
    --num_mini_batches 1 \
    --output_dir results_rloo \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --total_episodes 10000 \
    --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --missing_eos_penalty 1.0 \
    --response_length 256

accelerate launch --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
    examples/scripts/test_rloo_trl.py \
    --dataset_name cais/mmlu \
    --dataset_train_split test \
    --output_dir rloo_results \
    --rloo_k 2 \
    --num_ppo_epochs 1 \
    --num_mini_batches 1 \
    --learning_rate 3e-6 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --total_episodes 10000 \
    --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --local_rollout_forward_batch_size 1 \
    --missing_eos_penalty 1.0 \
    --response_length 256
"""


def craft_prompt(example):
    # Step-by-step instructions enforcing structured output and word limit
    instructions = (
        "You will answer a multiple-choice question. Follow these steps strictly:\n"
        "1. Analyze the question and all answer choices carefully.\n"
        "2. Think step-by-step and reason through the possible answers.\n"
        "   - Clearly explain your thought process inside <think> and </think> tags.\n"
        "3. Conclude your reasoning and select the most correct answer.\n"
        "4. **Word Limit:** Your entire response (reasoning + answer) must not exceed **300 words**.\n"
        "5. Output ONLY this format on the last line:\n"
        "   FINAL_ANSWER: <choice_idx>\n"
    )

    # Optional few-shot example (can be removed if unnecessary)
    few_shot = (
        "### Example ###\n"
        "Question: What is 5 + 7?\n"
        "Choices:\n"
        "0: 10\n"
        "1: 12\n"
        "2: 15\n"
        "3: 17\n\n"
        "<think> 5 plus 7 equals 12, which matches choice 1. </think>\n"
        "FINAL_ANSWER: 1\n\n"
        "################\n\n"
    )

    # Extract the question and choices
    question = example["question"]
    choices = example["choices"]
    choices_str = "\n".join([f"{i}: {choice}" for i, choice in enumerate(choices)])

    # Construct the full prompt
    prompt = (
        instructions + "\n" +
        few_shot +  # Remove this line if few-shot is not needed
        "### Your Turn ###\n"
        f"Question: {question}\n\n"
        f"Choices:\n{choices_str}\n\n"
    )

    return {"prompt": prompt}


if __name__ == "__main__":
    # Parse arguments from command line.
    parser = HfArgumentParser((ScriptArguments, RLOOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()
    
    # Remove the output directory if it already exists.
    shutil.rmtree(training_args.output_dir, ignore_errors=True)
    
    ################
    # Model & Tokenizer
    ################

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, padding_side="left", trust_remote_code=model_args.trust_remote_code
    )
    # Add a pad token if necessary.
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE

    # Load the SFT model for both policy and reference policy.
    ref_policy = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
    )
    policy = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
    )

    ################
    # Dataset
    ################
    # Load the dataset according to the script arguments.
    dataset = load_dataset(script_args.dataset_name, 'abstract_algebra')['test']
    # For example, we reserve the last 100 samples for evaluation.
    eval_samples = 10
    train_dataset = dataset.select(range(len(dataset) - eval_samples))
    eval_dataset = dataset.select(range(len(dataset) - eval_samples, len(dataset)))
    
    # craft dataset prompts
    train_dataset = dataset.map(craft_prompt, batched=False)
    eval_dataset = dataset.map(craft_prompt, batched=False)
    dataset_text_field = "prompt"  # Adjust this if your dataset uses another field name.

    def prepare_dataset(dataset, tokenizer):
        """Pre-tokenize the dataset before training (collation happens during training)."""
        def tokenize(element):
            outputs = tokenizer(element[dataset_text_field], padding=False)
            return {"input_ids": outputs["input_ids"]}
        return dataset.map(
            tokenize,
            batched=True,
            remove_columns=dataset.column_names,
            num_proc=training_args.dataset_num_proc,
        )

    # Preprocess the datasets only on the main process (for efficiency).
    with PartialState().local_main_process_first():
        train_dataset = prepare_dataset(train_dataset, tokenizer)
        eval_dataset = prepare_dataset(eval_dataset, tokenizer)

    ################
    # Training
    ################
           
    trainer = RLOOTrainer(
        config=training_args,
        processing_class=tokenizer,
        policy=policy,
        ref_policy=ref_policy,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()

    # Save the final model.
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)

    # Generate completions (for qualitative evaluation).
    trainer.generate_completions()
