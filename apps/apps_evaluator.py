# coding: utf-8

# Author: Du Mingzhe (mingzhe@nus.edu.sg)
# Date: 2025-03-19


import time
import json
import textwrap
from tqdm import tqdm
from typing import List
from tabulate import tabulate
from monolith import monolith
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

def apps_evaluation_unpacker(args):
    return AppsEvaluator.apps_evaluation(*args)

class AppsEvaluator:
    def __init__(self, monolith_retries, monolith_timeout, case_multiply, number_of_workers):
        self.monolith_retries = monolith_retries
        self.monolith_timeout = monolith_timeout
        self.case_multiply = case_multiply
        self.number_of_workers = number_of_workers
    
    @classmethod
    def apps_evaluation(cls, solution_code: str, test_cases: List, timeout: int) -> dict:
        response = {
            'passed': False, 
            'time': float('inf'), 
            'memory': float('inf'), 
            'integral': float('inf'), 
            'status': 'error',
        }
        
        try:            
            monolith_client = monolith.Monolith(backend_url='https://monolith.cool', retries=3)
            
            # Construct Test Code
            solution_code = textwrap.indent(solution_code.strip(), "\t")
            test_case_list_str = json.dumps(test_cases, indent=4)
            test_code = TEMPLATE.format(solution_code=solution_code, test_case_list=test_case_list_str, case_multiply=100)
            
            # Submit Test Code to Monolith
            task_id = monolith_client.post_code_submit(lang="python", libs=[], code=test_code, timeout=timeout, profiling=True)["task_id"]
            
            # Wait for Test Code to Finish
            for _ in range(timeout):
                time.sleep(2)
                result = monolith_client.get_code_result(task_id)
                if result["status"] != "processing":
                    break
            
            # Check if Test Code Passed
            response['status'] = result['status']

            # Update Response
            if result["status"] == "done":
                response['passed'] = True if result['output_dict']['stdout'] == 'Success\n' else False
                response['time'] = result['output_dict']['duration']
                response['memory'] = result['output_dict']['peak_memory']
                response['integral'] = result['output_dict']['integral']
        except Exception as e:
            response['status'] = 'error'
            print(f"Evaluation Error: {e}", e.with_traceback(None))
        finally:
            return response
    
    def apps_pipeline(self):
        monolith_client = monolith.Monolith(backend_url='https://monolith.cool', retries=3)
        
        for i in range(87, 100):
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
                
                # Prepare Test Packs (add code_prompt to each solution)
                test_packs = [(code_prompt + '\n' + solution, test_cases, self.monolith_timeout) for solution in solutions]
                
                print(f'[+] Problem {problem_id} [{index}/{total_count}] in [{i}% - {(i+1)}%] - {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')
                
                # Check Monolith Status
                while True:
                    queue_size = monolith_client.get_status().get('current_queue_size', -1)
                    if queue_size == 0:
                        break
                    time.sleep(5)
                
                # Parallel Evaluation
                results = list()
                with ThreadPool(self.number_of_workers) as pool:
                    with tqdm(total=len(test_packs), desc='Solution Evaluation') as pbar:
                        for result in pool.imap(apps_evaluation_unpacker, test_packs):
                            results.append(result)
                            pbar.update(1)
                
                # Prepare the table header
                headers = ["Passed", "Status", "Time (ms)", "Memory (kb)", "Integral (ms * kb)"]

                # Build the table rows based on your results
                table = []
                for result in results:
                    row = [
                        '🟢' if result['passed'] else '🔴',
                        result['status'],
                        f"{result['time']:.2f}",
                        f"{result['memory']:.2f}",
                        f"{str(result['integral'])}"
                    ]
                    table.append(row)

                # Print the formatted table
                print(tabulate(table, headers=headers, tablefmt="fancy_outline"))
                
                # Verify Solutions
                verified_solutions = list()
                for (solution, result) in zip(solutions, results):
                    verified_solutions.append({
                        'code': solution, 
                        'status': str(result['status']),
                        'passed': bool(result['passed']),
                        'time': float(result['time']),
                        'memory': float(result['memory']),
                        'integral': float(result['integral'])
                    })
                        
                # Save New APPS Data
                new_apps_data.append({
                    "problem_id": problem_id,
                    "problem_content": problem_content,
                    "code_prompt": code_prompt,
                    "difficulty": difficulty,
                    "solutions": json.dumps(verified_solutions),
                    "test_cases": json.dumps(test_cases),
                })
            
                print('============================================================')
            
            # Push New APPS Data to HF
            new_apps_dataset = Dataset.from_list(new_apps_data)
            new_apps_dataset.push_to_hub("Elfsong/APPS_New", 'verified', split=f"{i}_{(i+1)}")
            

if __name__ == "__main__":
    evaluator = AppsEvaluator(monolith_retries=3, monolith_timeout=90, case_multiply=1024, number_of_workers=48)
    evaluator.apps_pipeline()