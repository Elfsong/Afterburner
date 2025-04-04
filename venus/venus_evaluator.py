# coding: utf-8

# Author: Du Mingzhe (mingzhe@nus.edu.sg)
# Date: 2025-03-30


import time
import json
import random
import logging
import textwrap
import requests
from tqdm import tqdm
from tabulate import tabulate
from monolith import monolith
from typing import Any, List, final
from datasets import load_dataset, Dataset
from multiprocessing.dummy import Pool as ThreadPool

EVALUATION_TEMPLATE = """import io
import re
import itertools
import collections
import heapq
import bisect
import string
import sys
import functools
import math
import copy
import unittest

from math import floor, ceil, factorial, sqrt, inf
from sys import maxsize, stdin
from bisect import bisect_left, bisect_right
from itertools import permutations, zip_longest
from heapq import heappush, heappop, heapify
from collections import deque, defaultdict, OrderedDict
from typing import List, Optional, Tuple
from functools import lru_cache, cache


# Placeholder for the running solution.
def running_solution():
{solution_code}

class TestSolution(unittest.TestCase):
    def run_io_fun(self, input_data):
        backup_stdin = sys.stdin
        backup_stdout = sys.stdout
        try:
            sys.stdin = io.StringIO(input_data)
            captured_output = io.StringIO()
            sys.stdout = captured_output

            running_solution()

            captured_output.seek(0)
            return captured_output.read()
        finally:
            sys.stdin = backup_stdin
            sys.stdout = backup_stdout

def make_test_function(input_data, expected):
{test_case_evaluator}

    def test_method(self):
        actual = self.run_io_fun(input_data)
        passed = evaluate(expected, actual)
        self.assertTrue(passed)

    return test_method

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

GENERATION_TEMPLATE = """
## Instructions
You are an expert competitive programmer who excels at solving algorithm problems in multiple programming languages.
Your task is to implement a solution to the following problem in {target_lang}.

## Problem Description
{question}

## Starter Code
{wrap_code_block(target_lang, starter_code)}

## Output Format
- Provide the complete solution code in **one markdown code block** with appropriate language identifier.
- Implement the function with the exact signature (name, parameters, etc.) specified in the starter code.
- EXCLUDE ALL explanations, code comments, import/package/library statements, additional classes or functions outside of the starter code scope, or starting code like `if __name__ == "__main__":` or `func main()` or `package main` or `using namespace std;`.
- Use but do not redefine any helper data structures provided in the starter code (even if commented out).
"""

def venus_evaluation_unpacker(args):
    return VenusEvaluator.venus_evaluation(*args)

class VenusEvaluator:
    def __init__(self, lang, number_of_workers, case_multiply, monolith_timeout) -> None:
        self.lang = lang
        self.case_multiply = case_multiply
        self.monolith_timeout = monolith_timeout
        self.number_of_workers = number_of_workers
        self.monolith_client = monolith.Monolith(backend_url='https://monolith.cool', retries=3)

    @classmethod
    def venus_evaluation(cls, solution_code: str, instance: Any, case_multiply: int, timeout: int) -> dict:
        response = {'passed': False, 'time': float('inf'), 'memory': float('inf'), 'integral': float('inf'), 'status': 'error'}
        try:
            monolith_client = monolith.Monolith(backend_url='https://monolith.cool', retries=2)

            # Construct Test Code
            test_case_evaluator = instance['test_case_evaluator'].strip()
            test_case_runners = instance['test_case_runners']
            test_cases = json.loads(instance['test_cases'])

            solution_code = test_case_runners['python3'].replace('==Code Submission==', solution_code.strip())
            solution_code = textwrap.indent(solution_code.strip(), "    ")
            test_case_evaluator = textwrap.indent(test_case_evaluator, "    ")
            test_case_list_str = json.dumps(test_cases, indent=4)

            test_code = EVALUATION_TEMPLATE.format(solution_code=solution_code, test_case_evaluator=test_case_evaluator, test_case_list=test_case_list_str, case_multiply=case_multiply)

            # Submit Test Code to Monolith
            monolith_response = monolith_client.post_code_submit(lang="python", libs=[], code=test_code, timeout=timeout, profiling=True)
            task_id = monolith_response['task_id']
            if task_id is None:
                raise requests.exceptions.RequestException("Task ID is None" + "->" + str(monolith_response))
            
            # Wait for Test Code to Finish
            for _ in range(timeout):
                time.sleep(2)
                result = monolith_client.get_code_result(task_id)
                if result["status"] != "processing":
                    break
            
            # Check if Test Code Passed
            response['status'] = result['status']

            if result["status"] == "done":
                response['passed'] = True if result['output_dict']['stdout'] == 'Success\n' else False
                response['time'] = result['output_dict']['duration']
                response['memory'] = result['output_dict']['peak_memory']
                response['integral'] = result['output_dict']['integral']
        except Exception as e:
            print("Evaluation Error: ", e)
            response['status'] = 'error'
        finally:
            return response

    def venus_generation(self, instance: Any, target_lang: str) -> str:
        pass
    
    def venus_distribution_pipeline(self):
        # Load the datasets
        venus_dataset = load_dataset("Elfsong/Venus", "python3", split="train")

        venus_dict = dict()
        for instance in venus_dataset:
            problem_id = int(instance['question_id'])
            venus_dict[problem_id] = instance
        
        for i in range(67, 100):
            print(f'[+] Processing Range: [{i}% - {(i+1)}%]')
            leetcode_dataset = load_dataset("Elfsong/leetcode_data", split=f"train[{i}%:{(i+1)}%]")

            new_leetcode_data = list()

            for index, instance in enumerate(leetcode_dataset):
                problem_id = instance['problem_id']
                print(f"Processing Problem [{problem_id}]...")

                # Instance Check
                if not instance['test_case_runners']: continue
                if not instance['test_case_runners']['python3']: continue
                if not instance['test_case_evaluator']: continue

                # Prepare the solution code
                solution_list = list()

                # Example Solutions
                if 'solutions' in instance and self.lang in instance['solutions'] and instance['solutions'][self.lang]:
                    for solution in instance['solutions'][self.lang]:
                        solution_list.append(solution)
                
                # Human Solutions
                if problem_id in venus_dict:
                    rt_list = venus_dict[problem_id]['rt_list']
                    mm_list = venus_dict[problem_id]['mm_list']
                    for solution in rt_list + mm_list:
                        solution_list.append(solution['code'])

                # Solution Deduplication
                solution_list = list(set(solution_list))

                # Prepare Test Packs (add code_prompt to each solution)
                solution_list = random.sample(solution_list, min(500, len(solution_list))) # remove it later
                test_packs = [(solution, instance, self.case_multiply, self.monolith_timeout) for solution in solution_list]

                print(f'[+] Problem {problem_id} [{index}/{len(leetcode_dataset)}] in [{i}% - {(i+1)}%] - {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')
                    
                # Check Monolith Status
                while True:
                    monolith_queue_size = self.monolith_client.get_status().get('current_queue_size', -1)
                    if monolith_queue_size == 0:
                        break
                    time.sleep(5)

                # Parallel Evaluation
                results = list()
                with ThreadPool(self.number_of_workers) as pool:
                    with tqdm(total=len(test_packs), desc='Solution Evaluation') as pbar:
                        for result in pool.imap(venus_evaluation_unpacker, test_packs):
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
                for (solution, result) in zip(solution_list, results):
                    verified_solutions.append({
                        'code': solution, 
                        'status': str(result['status']),
                        'passed': bool(result['passed']),
                        'time': float(result['time']),
                        'memory': float(result['memory']),
                        'integral': float(result['integral'])
                    })
                
                # Save New APPS Data
                new_instance = {
                    "problem_id": int(instance['problem_id']),
                    "title": str(instance['title']),
                    "question_content": str(instance['question_content']),
                    "difficulty": str(instance['difficulty']),
                    "tags": list(instance['tags']),
                    "code_prompt": str(instance['code_prompt']['python3']),
                    "test_case_generator": str(instance['test_case_generator']),
                    "test_case_evaluator": str(instance['test_case_evaluator']),
                    "test_case_runners": str(instance['test_case_runners']["python3"]),
                    "test_cases": str(instance['test_cases']),
                    "solutions": verified_solutions
                }
                new_leetcode_data.append(new_instance)
            
                print('============================================================')

            # Push New Data to HF
            new_leetcode_dataset = Dataset.from_list(new_leetcode_data)
            new_leetcode_dataset.push_to_hub("Elfsong/Venus_python", 'verified', split=f"{i}_{(i+1)}", private=True)

    def venus_evalution_pipeline(self, model_name, k=1):
        # Load the datasets
        venus_dataset = load_dataset("Elfsong/Venus", "python3", split="train")
        

if __name__ == "__main__":
    venus_evaluator = VenusEvaluator(lang="python3", number_of_workers=64, case_multiply=64, monolith_timeout=90)
    venus_evaluator.venus_distribution_pipeline()