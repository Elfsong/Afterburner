# code: utf-8

# Author: Du Mingzhe (mingzhe@nus.edu.sg)
# Date: 2025-01-21

import sys
sys.path.append('/home/nus_cisco_wp1/Projects/llm-sandbox')

import json
import utils
from tqdm import tqdm
from random import sample
from llm_sandbox import SandboxSession
from datasets import load_dataset, Dataset

task_ds = load_dataset("Elfsong/Venus_t", "python3", split="eval")

for instance in tqdm(list(task_ds)):    
    try:
        # Extract instance details
        content = instance['content']
        libraries = instance['libraries']
        question_id = instance['question_id']
        entry_point = instance['entry_point']
        import_code = instance['import_code']
        solution_prompt = instance['code_prompt']
        test_cases = instance['test_case']
        
        if not test_cases: continue
        test_cases = instance['test_case'].split("\n")
        test_cases = [json.loads(test_case) for test_case in test_cases if test_case]
        test_case_generator = instance['test_case_generator']
        solutions = instance['rt_list'] + instance['mm_list']    
        solutions = [solution for solution in solutions if ('user.out' not in solution['code']) and ('Solution' in solution['code'])]

        # Inference Sampling
        
        # Evaluation

            
    except Exception as e:
        print(e)
        







