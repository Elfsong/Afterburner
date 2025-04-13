# coding: utf-8

# Author: Du Mingzhe
# Date: 2025-04-11
# Description: Prepare data for DPO training

import random
from tqdm import tqdm
from datasets import load_dataset, Dataset, DatasetDict

PROMPT_TEMPLATE = """
## Instructions
You are an expert competitive programmer who excels at solving algorithm problems in multiple programming languages.
Your task is to implement a solution to the following problem in {target_lang}.

## Problem Description
{question}

## Starter Code
{starter_code}

## Output Format
- Provide the complete solution in **one markdown code block** with the correct language identifier.
- Use the exact function signature (name, parameters, etc.) specified in the starter code.
- Make the code as {efficiency_instruction} as possible.
- Exclude all explanations, code comments, import/package/library statements, additional classes or functions outside the starter code scope, and any typical boilerplate like `if __name__ == "__main__":`, `func main()`, `package main`, or `using namespace std;`.
"""

def sft_construction_pipeline(data_split="100%", solution_num=8):
    ds = load_dataset("Elfsong/Venus_Python", split=f"train[:{data_split}]")

    time_solution_pairs = list()
    memory_solution_pairs = list()
    integral_solution_pairs = list()
    
    for instance in tqdm(ds, desc="Constructing SFT data"):
        solutions = [solution for solution in instance["solutions"] if solution["passed"]]
        
        # get the top-k solution
        top_time_solutions = sorted(solutions, key=lambda x: x["time"])[:solution_num]
        top_memory_solutions = sorted(solutions, key=lambda x: x["memory"])[:solution_num]
        top_integral_solutions = sorted(solutions, key=lambda x: x["integral"])[:solution_num]
        
        for solution in top_time_solutions:
            time_solution_pairs.append({
                "problem_id": instance["problem_id"],
                "type": "time",
                "prompt": PROMPT_TEMPLATE.format(target_lang="python3", question=instance["question_content"], starter_code=instance["code_prompt"], efficiency_instruction="time-efficient"),
                "response": solution["code"],
                "performance": float(solution["time"])
            })
        
        for solution in top_memory_solutions:
            memory_solution_pairs.append({
                "problem_id": instance["problem_id"],
                "type": "memory",
                "prompt": PROMPT_TEMPLATE.format(target_lang="python3", question=instance["question_content"], starter_code=instance["code_prompt"], efficiency_instruction="memory-efficient"),
                "response": solution["code"],
                "performance": float(solution["memory"])
            })
        
        for solution in top_integral_solutions:
            integral_solution_pairs.append({
                "problem_id": instance["problem_id"],
                "type": "integral",
                "prompt": PROMPT_TEMPLATE.format(target_lang="python3", question=instance["question_content"], starter_code=instance["code_prompt"], efficiency_instruction="integral-efficient (time + memory)"),
                "response": solution["code"],
                "performance": float(solution["integral"])
            })

    dd = DatasetDict({
        "time": Dataset.from_list(time_solution_pairs),
        "memory": Dataset.from_list(memory_solution_pairs),
        "integral": Dataset.from_list(integral_solution_pairs)
    })

    dd.push_to_hub("Elfsong/Venus_SFT_Data", private=True)

if __name__ == "__main__":
    sft_construction_pipeline(data_split="100%", solution_num=24)