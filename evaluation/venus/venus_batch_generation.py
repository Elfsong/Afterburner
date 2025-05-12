import sys
sys.path.append("/home/mingzhe/Projects/Afterburner")

import os
import json
import utils
import argparse
import anthropic
from tqdm import tqdm
from datasets import Dataset, load_dataset

parser = argparse.ArgumentParser(description='Process some parameters for Afterburner generation.')
parser.add_argument('--original_dataset_config', type=str, default="", help='Name of the original dataset config')
parser.add_argument('--afterburner_dataset_config', type=str, default="", help='Name of the afterburner dataset config')
parser.add_argument('--afterburner_model_name', type=str, default="", help='Name of the afterburner model')
parser.add_argument('--efficiency_instruction', type=str, default="integral", help='Efficiency instruction to optimize')
parser.add_argument('--inference_provider', type=str, default="openai", help='Inference provider')

args = parser.parse_args()

data_precentage = "100%"
original_dataset_config = args.original_dataset_config
afterburner_dataset_config = args.afterburner_dataset_config
afterburner_model_name = args.afterburner_model_name
efficiency_instruction = args.efficiency_instruction
inference_provider = args.inference_provider

AFTERBURNER_GENERATION_TEMPLATE = """
## Instructions
You are an expert competitive programmer who excels at solving algorithm problems in multiple programming languages.
Your task is to implement a solution to the following problem in {target_lang}.

## Problem Description
{problem_description}

## Original Solution
```python
{original_solution}
```

## Original Performance
Passed: {original_passed} / Time: {original_time} / Memory: {original_memory} / Integral: {original_integral}

## Output Format
- Provide the complete solution code in **one markdown code block** with appropriate language identifier.
- Fix the original solution if it was not passed. Optimize the {efficiency_instruction} performance if the original solution was passed.
- EXCLUDE ALL explanations, code comments, import/package/library statements, additional classes or functions outside of the starter code scope, or starting code like `if __name__ == "__main__":` or `func main()` or `package main` or `using namespace std;`.
"""

GENERATION_TEMPLATE = """
## Instructions
You are an expert competitive programmer who excels at solving algorithm problems in multiple programming languages.
Your task is to implement a solution to the following problem in {target_lang}.

## Problem Description
{question}

## Starter Code
{starter_code}

## Output Format
- Provide the complete solution code in **one markdown code block** with appropriate language identifier.
- Implement the function with the exact signature (name, parameters, etc.) specified in the starter code.
- EXCLUDE ALL explanations, code comments, import/package/library statements, additional classes or functions outside of the starter code scope, or starting code like `if __name__ == "__main__":` or `func main()` or `package main` or `using namespace std;`.

Your solution:
"""

# Generation Config
generation_config = f"{original_dataset_config}_{efficiency_instruction}_{afterburner_dataset_config}"

# Meta Dataset
venus_dataset = load_dataset("Elfsong/Venus_Python", split=f"test[:{data_precentage}]")

# Original Generation Dataset
if original_dataset_config:
    generation_dataset = load_dataset("Elfsong/Venus_Python_Model_Evaluation", original_dataset_config, split="train")
else:
    generation_dataset = None

batch_cases = list()
for instance in tqdm(venus_dataset, desc='Generating Solutions'):
    problem_id = instance['problem_id']
    
    if original_dataset_config:
        # Find matching solutions directly
        original_solutions = [s for s in generation_dataset if s['problem_id'] == int(instance['problem_id'])]
        original_solution_code = original_solutions[0]['solution_code'] if original_solutions else "No solution found"
        original_solution_passed = all(s['passed'] for s in original_solutions) if original_solutions else False
        original_solution_time = sum(s['absolute_time'] for s in original_solutions) / len(original_solutions) if original_solutions else float('inf')
        original_solution_memory = sum(s['absolute_memory'] for s in original_solutions) / len(original_solutions) if original_solutions else float('inf')
        original_solution_integral = sum(s['absolute_integral'] for s in original_solutions) / len(original_solutions) if original_solutions else float('inf')
    
        prompt = AFTERBURNER_GENERATION_TEMPLATE.format(
            target_lang="python",
            problem_description=instance['question_content'],
            original_solution=original_solution_code,
            original_passed=original_solution_passed,
            original_time=original_solution_time,
            original_memory=original_solution_memory,
            original_integral=original_solution_integral,
            efficiency_instruction=utils.EFFICIENCY_INSTRUCTIONS[efficiency_instruction]
        )
    else:
        prompt = GENERATION_TEMPLATE.format(
            target_lang="python",
            question=instance['question_content'],
            starter_code=instance['code_prompt']
        )
    
    if inference_provider == "openai":
        batch_case = {
            "custom_id": f"request-{problem_id}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": afterburner_model_name,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_completion_tokens": 8192*4
            }
        }
    elif inference_provider == "claude":
        batch_case = {
            "custom_id": f"request-{problem_id}",
            "params": {
                "model": afterburner_model_name,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 4096
            }
        }
    
    batch_cases.append(batch_case)
    

if inference_provider == "claude":
    client = anthropic.Anthropic(api_key=os.environ.get("CLAUDE_TOKEN"))
    message_batch = client.messages.batches.create(requests=batch_cases)

if inference_provider == "openai":
    with open(f"./batch_input/{generation_config}_batchinput.jsonl", "w") as f:
        for batch_case in batch_cases:
            f.write(json.dumps(batch_case) + "\n")