# coding: utf-8

# Author: Du Mingzhe (mingzhe@nus.edu.sg)
# Date: 2025-03-27

import os
import re
import time
import json
import logging
import textwrap
from tqdm import tqdm
from typing import Any
from openai import OpenAI
from monolith import monolith
from datasets import load_dataset
from multiprocessing.dummy import Pool as ThreadPool


# Logging Configuration
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# Console (stdout) handler
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# File handler (rotates at 5MB, keeps 3 backups)
file_handler = logging.FileHandler("benchmark.log")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

APPS_INFER_TEMPLATE = """
Problem:
{problem_content}

{code_prompt}

Write a complete, efficient Python program (compatible with Python 3.9+) that:
• Reads from standard input
• Writes only the required output (Provide only the source code.)
• Enclose the complete, executable solution between triple backticks (```python and ```).
"""

APPS_EVAL_TEMPLATE = """import io
import sys
import unittest

def solution():
{solution_code}

class TestSolution(unittest.TestCase):
    def run_io_fun(self, input_data):
        backup_stdin = sys.stdin
        backup_stdout = sys.stdout
        try:
            sys.stdin = io.StringIO(input_data)
            output_catcher = io.StringIO()
            sys.stdout = output_catcher

            solution()

            output_catcher.seek(0)
            return output_catcher.read()
        finally:
            sys.stdin = backup_stdin
            sys.stdout = backup_stdout

def make_test_function(input_data, expected):
    def test_function(self):
        actual = self.run_io_fun(input_data)
        self.assertEqual(expected, actual)
    return test_function

test_case_list = {test_case_list}
test_case_list = test_case_list * {case_multiply}

for i, case in enumerate(test_case_list, start=1):
    test_name = f"test_case_{{i}}"
    test_func = make_test_function(case['input'], case['output'])
    setattr(TestSolution, test_name, test_func)

if __name__ == '__main__':
    result = unittest.main(verbosity=2, exit=False)
    
    # If all tests passed, print "Success".
    if result.result.wasSuccessful():
        print("Success")
    else:
        print("Failed")
"""

class Benchmark:
    def __init__(self, dataset_name, model_name, case_multiply=1, enable_replay=False) -> None:
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.case_multiply = case_multiply
        self.enable_replay = enable_replay
        self.api_key = os.getenv('API_KEY')
        self.model_client = OpenAI(api_key=self.api_key)
        self.logger = logger.getChild(self.dataset_name)
        self.logger.info(f"[+] Initialized APPS Benchmark -> Dataset: [{dataset_name}] | Model: [{model_name}] | Case Multiply: [{case_multiply}] | Enable Replay: [{enable_replay}]")
        
    def inference(self, instance) -> Any:
        raise NotImplementedError("Method 'inference' must be implemented.")
    
    def instance_eval(self, instance) -> Any:
        raise NotImplementedError("Method 'instance_eval' must be implemented.")
    
    def eval(self) -> Any:
        raise NotImplementedError("Method 'eval' must be implemented.")
    
class APPSBenchmark(Benchmark):
    def __init__(self, dataset_name, model_name, case_multiply=1, enable_replay=False) -> None:
        super().__init__(dataset_name, model_name)
        
        # Load the dataset
        self.logger.info(f"[+] Loading dataset [{self.dataset_name}]...")
        self.ds = load_dataset("Elfsong/apps_verified", "default")['train']
        
    def instance_inference(self, instance) -> Any:        
        prompt = APPS_INFER_TEMPLATE.format(problem_content=instance['problem_content'], code_prompt=instance['code_prompt'])
        generated_code = None
        try:
            completion = self.model_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system",   "content": "You're a world-class programmer. Write an efficient Python solution to the given problem."},
                    {"role": "user",     "content": prompt},
                ]
            )
            response = completion.choices[0].message.content
            pattern = r"```python(.*?)```"
            generated_code = re.findall(pattern, response, flags=re.DOTALL)[0]
        except Exception as e:
            self.logger.error(f"[+] Generation Error: {e}")
        finally:
            return generated_code
    
    @classmethod
    def instance_eval(cls, solution_code, instance) -> Any:
        monolith_client = monolith.Monolith(backend_url='https://monolith.cool', retries=3)
        response = {'passed': False, 'time': 1e9, 'memory': 1e9, 'status': 'error', 'elapsed_time': 1e9}
        
        try:
            start_time = time.time()
            solution_code = textwrap.indent(solution_code.strip(), "\t")
            test_cases = json.loads(instance["test_cases"])
            test_case_list_str = json.dumps(test_cases, indent=4)
            test_code = APPS_EVAL_TEMPLATE.format(solution_code=solution_code, test_case_list=test_case_list_str, case_multiply=APPSBenchmark.case_multiply)
            
            # Submit Test Code to Monolith
            task_id = monolith_client.post_code_submit(lang="python", libs=[], code=test_code, timeout=60, profiling=False)["task_id"]
            
            # Wait for Test Code to Finish
            for _ in range(60):
                time.sleep(1)
                result = monolith_client.get_code_result(task_id)
                if result["status"] != "processing":
                    break
            
            # Check if Test Code Passed
            if result["status"] == "done":
                response['passed'] = True if result['output_dict']['stdout'] == 'Success\n' else False
                response['time'] = result['output_dict']['time_v']['elapsed_time_seconds']
                response['memory'] = result['output_dict']['time_v']['max_resident_set_kb']
            
            end_time = time.time()
            response['elapsed_time'] = end_time - start_time
            response['status'] = result['status']

        except Exception as e:
            print(f"Evaluation Error: {e}", e.with_traceback())
        finally:
            return response
    
    def eval(self) -> Any:
        def eval_wrapper(instance):
            problem_id = instance['problem_id']
            execution_result = None
            
            # Generating code solution for the problem
            generated_code = self.instance_inference(instance)
            if generated_code is None: return execution_result
                
            # Evaluating the generated code solution
            execution_result = APPSBenchmark.instance_eval(generated_code, instance)
            
            # Calculating the result distribution
            time_distribution, memory_distribution, integral_distribution = list(), list(), list()
            
            if APPSBenchmark.enabled_replay:
                for solution in instance['solutions']:
                    execution_result = APPSBenchmark.instance_eval(solution['code'], instance)
                    time_distribution.append(float(execution_result['time']))
                    memory_distribution.append(float(execution_result['memory']))
            else:
                for solution in instance['solutions']:
                    time_distribution.append(float(solution['time']) * APPSBenchmark.case_multiply)
                    memory_distribution.append(float(solution['memory']) * APPSBenchmark.case_multiply)
            
            result_str = json.dumps({
                "problem_id": problem_id,
                "passed": execution_result['passed'],
                "solution": generated_code,
                "execution_result": execution_result
            })
            
            self.logger.info(f"[+] Problem [{problem_id}] Result: {result_str}")
            return execution_result
        
        results = []
        with ThreadPool(16) as pool:
            with tqdm(total=len(self.ds), desc=f'Evaluating [{self.dataset_name}]') as pbar:
                for result in pool.imap(eval_wrapper, self.ds):
                    results.append(result)
                    pbar.update(1)
            
    
class MercuryBenchmark(Benchmark):
    def __init__(self, dataset_name, model_name) -> None:
        super().__init__(dataset_name, model_name)
        
        # Load the dataset
        self.logger.info(f"[+] Loading dataset [{self.dataset_name}]...")
        self.ds = load_dataset("Elfsong/Mercury", split="eval")
        
    def inference(self, instance) -> Any:
        return instance
    
    def instance_eval(self, instance) -> Any:
        return instance
    
    def eval(self) -> Any:
        return 0.0
    
if __name__ == '__main__':
    apps_benchmark = APPSBenchmark("apps_verified", 'o3-mini', case_multiply=512, enable_replay=False)
    apps_benchmark.eval()