import re 
import torch
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer

from grpo_01_reward_func import *


# Define system prompt
system_prompt = f"""
You are a mini-Sudoku solving assistant.

You will be given a 4x4 Sudoku puzzle where some cells are filled and others are empty.
The goal is to fill each empty cell with a number from 1 to 4 such that:

- Each row contains all numbers from 1 to 4 exactly once
- Each column contains all numbers from 1 to 4 exactly once
- Each 2x2 sub-grid contains all numbers from 1 to 4 exactly once

Think through the solution step by step.
Place your reasoning between {reasoning_start} and {reasoning_end}.
Then, provide your complete 4x4 solution grid between {solution_start} and {solution_end}.

The solution should be formatted as a 4x4 grid with spaces between numbers and newlines between rows.

For example:

{solution_start}
1 2 3 4
3 4 1 2
2 1 4 3
4 3 2 1
{solution_end}
"""

def preprocess_dataset(example):
    return {
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": parse_sudoku_question(example["question"])}
        ],
        "answer": example["answer"]
    }


train_dataset = ...
valid_dataset = ...

# completions = [[{
#     "content": system_prompt
# }]]
# print(reward_match_format_func(completions))
max_prompt_length = 100
max_seq_length = 256

traing_args = GRPOConfig(
    learning_rate = 5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    optim = "adamw_torch_fused",
    logging_steps = 10,
    per_device_train_batch_size = 4,
    gradient_accumulation_steps = 4,
    num_generations = 8,
    max_prompt_length = max_prompt_length,
    max_completion_length = max_seq_length - max_prompt_length,
    num_train_epochs = 4,
    save_steps = 100,
    report_to = "none",
    output_dir = "outputs",
)
processed_train_dataset = train_dataset.map(preprocess_dataset)
processed_validation_dataset = valid_dataset.map(preprocess_dataset)
trainer = GRPOTrainer(
    model = '/Qwen/Qwen3-0.6B',
    reward_func = [
        correctness_reward_func,
        reward_match_format_func,
        in_reward_func,
        grid_format_reward_func,
    ],
    train_dataset=processed_train_dataset,
    eval_dataset=processed_validation_dataset

)
trainer.train()
