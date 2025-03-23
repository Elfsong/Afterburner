# coding: utf-8

# Author: Du Mingzhe (mingzhe@nus.edu.sg)
# Date: 2025-03-19

import re
import os
import time
import json
import textwrap
from tqdm import tqdm
from typing import List
from openai import OpenAI
from monolith import monolith
# from multiprocessing import Pool
from collections import OrderedDict
from datasets import load_dataset, Dataset
from multiprocessing.dummy import Pool as ThreadPool


TEMPLATE = """import io
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

def apps_evaluation_unpacker(args):
    return AppsEvaluator.apps_evaluation(*args)

class AppsEvaluator:
    def __init__(self, model_name: str):
        # self.api_key = os.getenv('API_KEY')
        # self.model_client = OpenAI(api_key=self.api_key)
        self.model_name = model_name
        # self.number_of_workers = 8
    
    def model_inference(self, model_name: str, prompt: str) -> str:
        completion = self.model_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system",   "content": "You're a world-class programmer. Write an efficient Python solution to the given problem."},
                {"role": "user",     "content": prompt},
            ]
        )
        return completion.choices[0].message.content
    
    def apps_generation(self, problem_content: str, code_prompt: str) -> str:
        try:
            prompt = f"""
Problem:
{problem_content}

{code_prompt}

Instructions:
1. Write an efficient Python code for the problem.
2. Read input from stdin and output results to stdout.
3. Enclose the complete, executable solution between triple backticks (```python and ```).
            """
            response = self.model_inference(model_name=self.model_name, prompt=prompt)
            
            pattern = r"```python(.*?)```"
            generated_code = re.findall(pattern, response, flags=re.DOTALL)[0]
        except Exception as e:
            print(f"Generation Error: {e}")
            generated_code = ""
        return code_prompt+'\n'+generated_code
    
    @classmethod
    def apps_evaluation(cls, solution_code: str, test_cases: List, timeout: int) -> bool:
        response = {'passed': False, 'time': 1e9, 'memory': 1e9, 'status': 'error', 'elapsed_time': 1e9, 'wait_time': 1e9, 'process_time': 1e9}
        try:
            start_time = time.time()
            
            monolith_client = monolith.Monolith(backend_url='https://monolith.cool')

            try:
                while True:
                    status = monolith_client.get_status()
                    ratio = status['current_queue_size'] / status['max_queue_size']
                    if ratio < 0.8:
                        break
                    else:
                        print(f"Monolith Queue Full: {ratio}")
                        time.sleep(5)
            except Exception as e:
                print(f"Monolith Busy: {e}")
                time.sleep(5)
            
            # Construct Test Code
            solution_code = textwrap.indent(solution_code.strip(), "\t")
            test_case_list_str = json.dumps(test_cases, indent=4)
            test_code = TEMPLATE.format(solution_code=solution_code, test_case_list=test_case_list_str)
            
            # Submit Test Code to Monolith
            task_id = monolith_client.post_code_submit(lang="python", libs=[], code=test_code, timeout=timeout, profiling=False)["task_id"]
            
            # Wait for Test Code to Finish
            start_time_1 = time.time()
            for _ in range(timeout):
                time.sleep(1)
                result = monolith_client.get_code_result(task_id)
                if result["status"] != "processing":
                    break
            end_time_1 = time.time()
            
            # Check if Test Code Passed
            if result["status"] == "done":
                response['passed'] = True if result['output_dict']['stdout'] == 'Success\n' else False
                response['time'] = result['output_dict']['time_v']['elapsed_time_seconds']
                response['memory'] = result['output_dict']['time_v']['max_resident_set_kb']
            
            end_time = time.time()
            response['elapsed_time'] = end_time - start_time
            response['wait_time'] = end_time_1 - start_time_1
            response['status'] = result['status']

        except Exception as e:
            print(f"Evaluation Error: {e}")
        finally:
            return response
    
    def apps_pipeline(self):
        for i in range(85, 100):
            print(f'[+] Processing Test Set: {i}% - {(i+1)}%')
            apps_data = load_dataset("Elfsong/APPS", 'default', split=f"test[{i}%:{(i+1)}%]")        
            new_apps_data = list()
            
            total_count = len(apps_data)
            for index, instance in enumerate(apps_data):
                problem_id = instance["problem_id"]
                problem_content = instance["question"]
                code_prompt = instance["starter_code"]
                difficulty = instance["difficulty"]
                solutions = json.loads(instance["solutions"])
                test_cases = json.loads(instance["test_cases"])
                print(f'[+] Problem-{problem_id} [{index}/{total_count}]')
                
                cases = [(solution, test_cases, 60) for solution in solutions]
                
                # Check Solutions
                results = list()
                
                with ThreadPool(64) as pool:
                    with tqdm(total=len(solutions), desc='Solution Evaluation') as pbar:
                        for result in pool.imap(apps_evaluation_unpacker, cases):
                            results.append(result)
                            pbar.update(1)

                # Brief Results
                # print('Results:', ''.join('🟢' if result['passed'] else '🔴' for result in results))
                
                # Detailed Results
                for result in results:
                    print(f"[-] Passed: {'🟢' if result['passed'] else '🔴'} \t Time: {result['time']:.2f} Sec \t Status: {result['status']} \t Elapsed_time: {result['elapsed_time']:.2f} \t Wait_time: {result['wait_time']:.2f}")
                
                verified_solutions = list()
                for (solution, result) in zip(solutions, results):
                    if result['passed']:
                        verified_solutions.append({
                            'code': solution, 
                            'passed': bool(result['passed']),
                            'time': float(result['time']),
                            'memory': float(result['memory']),
                            'status': str(result['status'])
                        })
                
                new_apps_data.append({
                    "problem_id": problem_id,
                    "problem_content": problem_content,
                    "code_prompt": code_prompt,
                    "difficulty": difficulty,
                    "solutions": json.dumps(verified_solutions),
                    "test_cases": json.dumps(test_cases),
                })
            
                
                # # Generate Solution Code
                # generated_solution = self.apps_generation(problem_content=problem_content, code_prompt=code_prompt)
                
                # # Evaluate Solution Code
                # passed = self.apps_evaluation(solution_code=generated_solution, test_cases=test_cases, timeout=60)
                # print(f"Problem ID: {problem_id}, Passed: {passed}")
                
                # if passed:
                #     passed_count += 1
                # total_count += 1
                print('============================================================')
            
            # Save New APPS Data
            new_apps_dataset = Dataset.from_list(new_apps_data)
            new_apps_dataset.push_to_hub("Elfsong/APPS", 'verified', split=f"{i}_{(i+1)}")
            
            # print(f"Total: {total_count}, Passed: {passed_count}, Pass@1: {passed_count/total_count:.2f}")
            

if __name__ == "__main__":
    evaluator = AppsEvaluator(model_name="gpt-4o")
    evaluator.apps_pipeline()