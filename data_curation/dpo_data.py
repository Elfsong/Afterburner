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

def dpo_construction_pipeline(data_split="100%", solution_num=8):
    ds = load_dataset("Elfsong/Venus_Python", split=f"train[:{data_split}]")

    time_solution_pairs = list()
    memory_solution_pairs = list()
    integral_solution_pairs = list()
    
    for instance in tqdm(ds, desc="Constructing DPO data"):
        solutions = [solution for solution in instance["solutions"] if solution["passed"]]
        if len(solutions) < solution_num: continue

        for _ in range(solution_num):
            # Randomly select TWO solutions
            selected_solutions = random.sample(solutions, 2)
  
            # Sort by time
            selected_solutions_time = sorted(selected_solutions, key=lambda x: x["time"])
            time_solution_pairs.append({
                "problem_id": instance["problem_id"],
                "type": "time",
                "prompt": PROMPT_TEMPLATE.format(target_lang="python3", question=instance["question_content"], starter_code=instance["code_prompt"], efficiency_instruction="time-efficient"),
                "chosen": selected_solutions_time[0]["code"],
                "rejected": selected_solutions_time[1]["code"],
                "chosen_performance": float(selected_solutions_time[0]["time"]),
                "rejected_performance": float(selected_solutions_time[1]["time"])
            })

            # Sort by memory
            selected_solutions_memory = sorted(selected_solutions, key=lambda x: x["memory"])
            memory_solution_pairs.append({
                "problem_id": instance["problem_id"],
                "type": "memory",
                "prompt": PROMPT_TEMPLATE.format(target_lang="python3", question=instance["question_content"], starter_code=instance["code_prompt"], efficiency_instruction="memory-efficient"),
                "chosen": selected_solutions_memory[0]["code"],
                "rejected": selected_solutions_memory[1]["code"],
                "chosen_performance": float(selected_solutions_memory[0]["memory"]),
                "rejected_performance": float(selected_solutions_memory[1]["memory"])
            })

            # Sort by integral
            selected_solutions_integral = sorted(selected_solutions, key=lambda x: x["integral"])    
            integral_solution_pairs.append({
                "problem_id": instance["problem_id"],
                "type": "integral",
                "prompt": PROMPT_TEMPLATE.format(target_lang="python3", question=instance["question_content"], starter_code=instance["code_prompt"], efficiency_instruction="integral-efficient (time + memory)"),
                "chosen": selected_solutions_integral[0]["code"],
                "rejected": selected_solutions_integral[1]["code"],
                "chosen_performance": float(selected_solutions_integral[0]["integral"]),
                "rejected_performance": float(selected_solutions_integral[1]["integral"])
            })

    dd = DatasetDict({
        "time": Dataset.from_list(time_solution_pairs),
        "memory": Dataset.from_list(memory_solution_pairs),
        "integral": Dataset.from_list(integral_solution_pairs)
    })

    dd.push_to_hub("Elfsong/Venus_DPO_Data", private=True)

if __name__ == "__main__":
    dpo_construction_pipeline(data_split="100%", solution_num=48)