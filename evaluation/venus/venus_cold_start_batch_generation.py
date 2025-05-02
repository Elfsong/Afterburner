import sys
sys.path.append("/home/mingzhe/Projects/Afterburner")

import os
import re
import json
import utils
import random
import argparse
import anthropic
from tqdm import tqdm
import multiprocessing
from google import genai
from google.genai import types
from datasets import Dataset, load_dataset
from multiprocessing.dummy import Pool as ThreadPool

SYSTEM_PROMPT = """
A conversation between User and Assistant. The user asks a question and provides an original solution, then the Assistant improve it. 
The assistant first thinks about the reasoning process in the mind and then provides the user with the improved solution. 
The reasoning process and solution are enclosed within <thinking> </thinking> and <solution> </solution> tags, respectively.
For example, "<thinking>reasoning_process</thinking><solution>improved_solution</solution>".
"""

AFTERBURNER_GENERATION_TEMPLATE = """
## Instructions
You are an expert competitive programmer skilled in solving algorithmic challenges efficiently across various programming languages. 
Your goal is to critically evaluate and enhance the provided solution in {target_lang}.

## Problem Description
{problem_description}

## Original Solution
```python
{original_solution}
```

## Original Performance
Passed Test Cases: {original_passed} / Execution Time: {original_time} / Memory Usage: {original_memory} / Integral of Time and Memory: {original_integral}

## Task
- If the original solution did not pass all test cases, identify and correct the errors.
- If the original solution passed test cases, optimize the solution for {efficiency_instruction}.

## Output Requirements
- Begin your response with your thought process clearly enclosed within <thinking>...</thinking> tags.
- Present your improved solution enclosed within <solution>...</solution> tags.
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

# Meta Dataset
venus_dataset = load_dataset("Elfsong/Venus_Python", split=f"train[:100%]")

batch_cases = list()
for efficiency_instruction in utils.EFFICIENCY_INSTRUCTIONS:
    for instance in tqdm(venus_dataset, desc=f'Generating data for {efficiency_instruction}...'):
        problem_id = instance['problem_id']
        problem_description = instance['question_content']
        solutions = instance['solutions']

        for solution in solutions:
            solution_code = solution['code']
            solution_passed = solution['passed']
            solution_time = solution['time']
            solution_memory = solution['memory']
            solution_integral = solution['integral']

            prompt = AFTERBURNER_GENERATION_TEMPLATE.format(
                target_lang="python", 
                problem_description=problem_description,
                original_solution=solution_code,
                original_passed=solution_passed, 
                original_time=solution_time,
                original_memory=solution_memory, 
                original_integral=solution_integral, 
                efficiency_instruction=efficiency_instruction
            )

            batch_cases.append({
                "problem_id": problem_id,
                "prompt": prompt,
                "solution": solution_code
            })
print(f"[+] Generated {len(batch_cases)} cases.")

client = genai.Client(api_key="API_TOKEN")

def process_batch_case(batch_case):
    response_text, thinking, solution_code = None, None, None
    
    try:
        # Generate content with the model
        response = client.models.generate_content(
            model="gemini-2.5-pro-preview-03-25",
            contents=SYSTEM_PROMPT + batch_case["prompt"],
            config=types.GenerateContentConfig(thinking_config=types.ThinkingConfig(thinking_budget=4096))
        )
        
        # Extract data from response
        response_text = response.text
        
        # Use more efficient pattern matching with compiled regex patterns
        thinking_match = re.search(r'<thinking>(.*?)</thinking>', response_text, re.DOTALL)
        thinking = thinking_match.group(1) if thinking_match else None
        
        solution_match = re.search(r'<solution>(.*?)</solution>', response_text, re.DOTALL)
        if solution_match:
            solution = solution_match.group(1)
            code_blocks = utils.extract_code_blocks(solution)
            solution_code = code_blocks[-1]["code"] if code_blocks else None
            
    except Exception as e:
        print(f"Error processing batch case {batch_case['problem_id']}: {e}")
    
    # Return result outside of try/finally for cleaner flow
    return {
        "problem_id": batch_case["problem_id"],
        "prompt": batch_case["prompt"], 
        "solution": batch_case["solution"],
        "response": response_text,
        "thinking": thinking,
        "solution_code": solution_code
    }

for i in range(100):
    print(f"[+] Generating batch {i}...")
    batch_cases = random.sample(batch_cases, 128)
    print(f"[+] Sampling {len(batch_cases)} cases.")
    
    results = list()
    with ThreadPool(32) as pool:
        with tqdm(total=len(batch_cases), desc='Solution Evaluation') as pbar:
            for result in pool.imap(lambda batch_case: process_batch_case(batch_case), batch_cases):
                if result is not None:
                    results.append(result)
                pbar.update(1)

    ds = Dataset.from_list(results)
    ds.push_to_hub("Elfsong/venus_cold_start", f'batch_new_big_batch_{i}', private=True)