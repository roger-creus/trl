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

from IPython import embed

"""
python examples/scripts/test_rloo_trl.py \
    --dataset_name cais/mmlu \
    --dataset_train_split test \
    --learning_rate 3e-6 \
    --num_ppo_epochs 1 \
    --num_mini_batches 1 \
    --output_dir results_rloo \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --total_episodes 10000 \
    --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --temperature 0.6 \
    --stop_token eos

accelerate launch --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
    examples/scripts/test_rloo_trl.py \
    --dataset_name cais/mmlu \
    --dataset_train_split test \
    --output_dir rloo_results \
    --rloo_k 4 \
    --local_rollout_forward_batch_size 4 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --num_ppo_epochs 1 \
    --num_mini_batches 1 \
    --learning_rate 3e-6 \
    --total_episodes 10000 \
    --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --temperature 0.6 \
    --response_length 512 \
    --stop_token eos
"""


def craft_prompt(example):
    # Extract question and choices from the example.
    question = example["question"]
    choices = example["choices"]

    # Create a string with numbered choices.
    choices_str = "\n".join([f"{i}: {choice}" for i, choice in enumerate(choices)])

    # Few-shot examples to guide the model.
    few_shot_examples = (
        "Example 1:\n"
        "Question: What is 3 * 3?\n"
        "Choices:\n"
        "0: 6\n"
        "1: 9\n"
        "2: 12\n"
        "3: 15\n"
        "ANSWER: 1\n\n"
        "Example 2:\n"
        "Question: What is 7 + 5?\n"
        "Choices:\n"
        "0: 10\n"
        "1: 13\n"
        "2: 12\n"
        "3: 14\n"
        "ANSWER: 2\n\n"
        "Example 3:\n"
        "Question: What is 10 - 5?\n"
        "Choices:\n"
        "0: 4\n"
        "1: 6\n"
        "2: 7\n"
        "3: 5\n"
        "ANSWER: 3\n\n"
    )
    
    # System prompt instructs the model about the expected answer format.
    system_prompt = (
        "Answer the following multiple-choice question by providing 'ANSWER: <integer_index_of_the_correct_choice>' "
        "Always conclude your response with 'ANSWER: <integer_idx>' with no additional contents. "
        f"You have a budget of {training_args.response_length // 2} words to generate the answer.\n\n" 
        + few_shot_examples
    )

    # User prompt provides the actual question and its choices.
    user_prompt = (
        f"Question: {question}\n\n"
        f"Choices:\n{choices_str}\n\n"
    )

    # Return the conversation in a list of messages.
    return {"prompt": system_prompt + "### Your turn ###\n" + user_prompt}


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
