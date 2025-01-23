# code: utf-8

# Author: Du Mingzhe (mingzhe@nus.edu.sg)
# Date: 2025-01-23

import sys
sys.path.append('/home/nus_cisco_wp1/Projects/llm-sandbox')
sys.path.append('/home/nus_cisco_wp1/Projects/Afterburner')

import json
import utils
from tqdm import tqdm
from llm_sandbox import SandboxSession
from datasets import load_dataset, Dataset

task_num = 128
temperature = 0.7
batch_number, batch_size = 8, 4
model_path = "microsoft/Phi-3.5-mini-instruct"
model_name = "Phi_3_5_mini_instruct_vanilla"

def code_construct(code, input_data):
    code_template = """
import sys
from io import StringIO

mock_input = StringIO('''{input_data}''')
sys.stdin = mock_input

{code}

sys.stdin = sys.__stdin__
    """
    return code_template.format(code=code, input_data=input_data)

task_ds = load_dataset("codeparrot/apps", "competition", split='test')
solution_ds = load_dataset("Elfsong/apps_generation", f'{model_name}_temperature_{temperature}')

tasks = dict()
for instance in task_ds:
    tasks[instance['problem_id']] = instance

for problem_id in tqdm(solution_ds):
    solutions = solution_ds[problem_id]
    instance = tasks[int(problem_id)]
    # oracle_solutions = json.loads(instance['solutions'])
    test_cases = json.loads(instance['input_output'])
    
    # # Oracle solution
    # with SandboxSession(lang="python", verbose=False) as session:
    #     print("\nOracle Solution:", end='')
    #     for input_data, output_data in zip(test_cases['inputs'], test_cases['outputs']):
    #         exec_code = code_construct(oracle_solutions[0], input_data)
    #         response = utils.run_with_timeout(session.run, 60, code=exec_code.strip(), run_memory_profile=True)
    #         if response['output']['stdout'] and response['output']['stdout'].strip() == output_data.strip():
    #             print("🟢", end='')
    #         else:
    #             print("🔴", end='')
    #             break
    
    # Generated Solutions
    for index, solution in enumerate(solutions):
        code = solution['code']
        problem_id = solution['problem_id']
        
        pass_ = True
        runtime_ = 0
        peak_memory = 0
        integral = 0
        
        # Test cases
        with SandboxSession(lang="python", verbose=False) as session:
            print(f"\nSolution {index}: ", end='')
            for input_data, output_data in zip(test_cases['inputs'], test_cases['outputs']):
                exec_code = code_construct(code, input_data)
                # session.setup(libraries=instance['libraries'])
                response = utils.run_with_timeout(session.run, 60, code=exec_code.strip(), run_memory_profile=True)
                if not response['error'] and response['output']['stdout'] and response['output']['stdout'].strip() == output_data.strip():
                    print("🟢", end='')
                    runtime_ += response['output']['duration']
                    peak_memory = max(peak_memory, response['output']['peak_memory'])
                    integral += response['output']['integral']
                else:
                    print("🔴", end='')
                    pass_ = False
                    break
                    
                
                

        

