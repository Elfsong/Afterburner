# coding: utf-8
# Author: Mingzhe Du (mingzhe@nus.edu.sg)
# Date: 2025-04-29

import os
import re
import random
import datasets

SYSTEM_PROMPT = """
A conversation between User and Assistant. The user asks a question and provides an original solution, then the Assistant improve it. 
The assistant first thinks about the reasoning process in the mind and then provides the user with the improved solution. 
The reasoning process and solution are enclosed within <thinking> </thinking> and <solution> </solution> tags, respectively.
For example, "<thinking>reasoning_process</thinking><solution>improved_solution</solution>".
"""

AFTERBURNER_TEMPLATE = """
## Instructions
You are an expert competitive programmer who excels at solving algorithm problems in multiple programming languages.
Your task is to implement a solution to the following problem in {target_lang}.
## Problem Description
{problem_description}
## Original Solution
{original_solution}
## Original Performance
Passed: {original_passed} / Time: {original_time} / Memory: {original_memory} / Integral: {original_integral}
## Output Format
- Provide the complete solution code in **one markdown code block** with appropriate language identifier.
- Fix the original solution if it was not passed. Optimize the {efficiency_instruction} performance if the original solution was passed.
- EXCLUDE ALL explanations, code comments, import/package/library statements, additional classes or functions outside of the starter code scope, or starting code like `if __name__ == "__main__":` or `func main()` or `package main` or `using namespace std;`.
"""

EFFICIENCY_INSTRUCTIONS = {
    "time": "time efficient",
    "memory": "memory efficient",
    "integral": "both time and memory efficient",
}

if __name__ == "__main__":
    sample_num = 1
    local_dir = "~/data/venus"
    data_source = "Elfsong/Venus_Python"
    dataset = datasets.load_dataset(data_source)
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    # add a row to each data item that represents a unique id
    def make_map_fn(split, efficiency_instruction):
        def process_fn(example, efficiency_instruction=efficiency_instruction):
            solutions = example['solutions']
            original_solution = random.choice(solutions)
            efficiency_instruction_str = EFFICIENCY_INSTRUCTIONS[efficiency_instruction]

            afterburner_prompt = AFTERBURNER_TEMPLATE.format(
                target_lang="python",
                problem_description=example['question_content'],
                efficiency_instruction=efficiency_instruction_str,
                original_solution=original_solution['code'],
                original_passed=original_solution['passed'],
                original_time=original_solution['time'],
                original_memory=original_solution['memory'],
                original_integral=original_solution['integral'],
            )

            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": afterburner_prompt,
                    }
                ],
                "ability": "code",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": original_solution
                },
                "extra_info": {
                    "split": split,
                    "problem_id": example['problem_id'],
                    "efficiency_instruction": efficiency_instruction,
                    "instance": example,
                    "case_multiply": 64
                },
            }
            return data

        return process_fn

    # Map for time optimization
    train_time_datasets = [train_dataset.map(function=make_map_fn("train", efficiency_instruction="time"), with_indices=False) for _ in range(sample_num)]
    test_time_datasets = [test_dataset.map(function=make_map_fn("test", efficiency_instruction="time"), with_indices=False) for _ in range(sample_num)]
    
    # Map for memory optimization 
    train_memory_datasets = [train_dataset.map(function=make_map_fn("train", efficiency_instruction="memory"), with_indices=False) for _ in range(sample_num)]
    test_memory_datasets = [test_dataset.map(function=make_map_fn("test", efficiency_instruction="memory"), with_indices=False) for _ in range(sample_num)]
    
    # Map for integral optimization
    train_integral_datasets = [train_dataset.map(function=make_map_fn("train", efficiency_instruction="integral"), with_indices=False) for _ in range(sample_num)]
    test_integral_datasets = [test_dataset.map(function=make_map_fn("test", efficiency_instruction="integral"), with_indices=False) for _ in range(sample_num)]

    train_datasets = train_time_datasets + train_memory_datasets + train_integral_datasets
    test_datasets = test_time_datasets + test_memory_datasets + test_integral_datasets

    train_dataset = datasets.concatenate_datasets(train_datasets)
    test_dataset = datasets.concatenate_datasets(test_datasets)
    
    train_dataset.to_parquet(os.path.join(local_dir, "venus_train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "venus_test.parquet"))
