# coding: utf-8

# Author: Du Mingzhe (mingzhe@nus.edu.sg)
# Date: 2025-03-30


import time
import json
import logging
import textwrap
from tqdm import tqdm
from tabulate import tabulate
from monolith import monolith
from typing import Any, List, final
from datasets import load_dataset, Dataset
from multiprocessing.dummy import Pool as ThreadPool

TEMPLATE = """import io
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

def venus_evaluation_unpacker(args):
    return VenusEvaluator.venus_evaluation(*args)

class VenusEvaluator:
    def __init__(self, lang, number_of_workers, case_multiply, monolith_timeout) -> None:
        self.lang = lang
        self.case_multiply = case_multiply
        self.monolith_timeout = monolith_timeout
        self.number_of_workers = number_of_workers
        

    @classmethod
    def venus_evaluation(cls, solution_code: str, instance: Any, case_multiply: int, timeout: int) -> dict:
        response = {'passed': False, 'time': float('inf'), 'memory': float('inf'), 'integral': float('inf'), 'status': 'error'}
        try:
            monolith_client = monolith.Monolith(backend_url='https://monolith.cool', retries=3)

            # Construct Test Code
            test_case_evaluator = instance['test_case_evaluator'].strip()
            test_case_runners = instance['test_case_runners']
            test_cases = json.loads(instance['test_cases'])

            solution_code = test_case_runners['python3'].replace('==Code Submission==', solution_code.strip())
            solution_code = textwrap.indent(solution_code.strip(), "    ")
            test_case_evaluator = textwrap.indent(test_case_evaluator, "    ")
            test_case_list_str = json.dumps(test_cases, indent=4)

            test_code = TEMPLATE.format(solution_code=solution_code, test_case_evaluator=test_case_evaluator, test_case_list=test_case_list_str, case_multiply=case_multiply)

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

            if result["status"] == "done":
                response['passed'] = True if result['output_dict']['stdout'] == 'Success\n' else False
                response['time'] = result['output_dict']['duration']
                response['memory'] = result['output_dict']['peak_memory']
                response['integral'] = result['output_dict']['integral']
            
        except Exception as e:
            print("Evaluation Error: ", e)
            print(logging.exception(e))
            response['status'] = 'error'
        finally:
            return response

    def venus_pipeline(self):
        monolith_client = monolith.Monolith(backend_url='https://monolith.cool', retries=3)

        # Load the datasets
        venus_dataset = load_dataset("Elfsong/Venus", "python3", split="train")

        venus_dict = dict()
        for instance in venus_dataset:
            problem_id = int(instance['question_id'])
            venus_dict[problem_id] = instance
        
        for i in range(100):
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

                # Prepare Test Packs (add code_prompt to each solution)
                test_packs = [(solution, instance, self.case_multiply, self.monolith_timeout) for solution in solution_list]

                print(f'[+] Problem {problem_id} [{index}/{len(leetcode_dataset)}] in [{i}% - {(i+1)}%] - {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')
                    
                # Check Monolith Status
                while True:
                    monolith_queue_size = monolith_client.get_status().get('current_queue_size', -1)
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
                instance['verified_solutions'] = dict()
                instance['verified_solutions'][self.lang] = verified_solutions
                new_leetcode_data.append(instance)
            
                print('============================================================')

            # Push New Data to HF
            new_leetcode_dataset = Dataset.from_list(new_leetcode_data)
            new_leetcode_dataset.push_to_hub("Elfsong/Venus_python", 'verified', split=f"{i}_{(i+1)}")


if __name__ == "__main__":
    venus_evaluator = VenusEvaluator(lang="python3", number_of_workers=48, case_multiply=64, monolith_timeout=90)
    venus_evaluator.venus_pipeline()