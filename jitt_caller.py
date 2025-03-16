# coding: utf-8

# Just-In-Time TestER (JITT)
# Author: Du Mingzhe (mingzhe@nus.edu.sg)
# Date: 2025-03-16

import time
import random
from tqdm import tqdm
from jitt import JITT
from datasets import load_dataset

# Load Venus Datasets
ds = load_dataset("Elfsong/Venus", "python3")

# Initialize JITT
jitt = JITT(number_of_cases=100, timeout=60)

t, c = 0, 0
for instance in tqdm(ds['train']):
    try:
        solutions = instance['rt_list'] + instance['mm_list']
        for solution in random.sample(solutions, min(jitt.number_of_samples, len(solutions))):
            solution_code = solution['code']
            
            generator_code = jitt.python_test_case_generator(solution_code)
            test_case_construction = jitt.python_test_case_construction(solution_code, generator_code)
            if not test_case_construction: continue
            
            # Submit code to Monolith
            task_id = jitt.post_code_submit(test_case_construction['libraries'], test_case_construction['executable_code'], timeout=jitt.timeout, profiling=False)
            
            # Retrieve test cases
            response = {'status': 'pending'}
            for i in tqdm(range(jitt.timeout), desc='Waiting for test case generation...'):
                time.sleep(1)
                response = jitt.get_code_result(task_id)
                if response['status'] in ['done', 'error', 'timeout']:
                    break
                
            if response['status'] == 'done':
                test_cases = jitt.test_case_construction(response['output_dict']['stdout'])
                print(test_cases)
                c += 1
    except Exception as e:
        pass
    finally:
        t += 1
        print(f"Total: {t}, Completed: {c}")