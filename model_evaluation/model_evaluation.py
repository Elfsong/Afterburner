# code: utf-8

# Author: Du Mingzhe (mingzhe@nus.edu.sg)
# Date: 2025-01-23

import json
from datasets import load_dataset, Dataset

task_num = 128
temperature = 0.7
batch_number, batch_size = 8, 4
model_path = "microsoft/Phi-3.5-mini-instruct"
model_name = "Phi_3_5_mini_instruct_vanilla"

task_ds = load_dataset("codeparrot/apps", "competition", split='test')

solution_ds = load_dataset("Elfsong/apps_generation", f'{model_name}_temperature_{temperature}')

def code_construct(code, input_data):
    code_template = """
import sys
from io import StringIO

mock_input = StringIO({input_data})
sys.stdin = mock_input

{code}

sys.stdin = sys.__stdin__
    """
    return code_template.format(code=code, input_data=input_data)

tasks = dict()
for instance in task_ds:
    tasks[instance['problem_id']] = instance

for problem_id in solution_ds:
    solutions = solution_ds[problem_id]
    for solution in solutions:
        code = solution['code']
        problem_id = solution['problem_id']
        test_cases = json.loads(instance['input_output'])
        instance = tasks[problem_id]
        
        for input_data, output_data in zip(test_cases['input'], test_cases['output']):
            exec_code = code_construct(code, input_data)
            exec(exec_code)
            assert exec_code == output_data

        

