# coding: utf-8

# Author: Du Mingzhe (mingzhe@nus.edu.sg)
# Date: 2025-03-30


import time
import json
import textwrap
import autoimport
from tqdm import tqdm
from tabulate import tabulate
from monolith import monolith
from typing import Any, List, final
from datasets import load_dataset, Dataset
from multiprocessing.dummy import Pool as ThreadPool

TEMPLATE = """import io
import sys
import unittest


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

class VenusEvaluator:
    def __init__(self, lang) -> None:
        self.lang = lang

    @classmethod
    def venus_evaluation(cls, solution_code: str, instance: Any, timeout: int) -> dict:
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

            test_code = TEMPLATE.format(solution_code=solution_code, test_case_evaluator=test_case_evaluator, test_case_list=test_case_list_str, case_multiply=100)
            test_code = autoimport.fix_code(test_code)

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
            response['status'] = 'error'
        finally:
            return response

    def venus_pipeline(self):
        # Load the dataset
        leetcode_dataset = load_dataset("Elfsong/leetcode_data", split="train")

        for instance in tqdm(leetcode_dataset):
            problem_id = instance['problem_id']
            print(f"Processing Problem [{problem_id}]...")

            # Prepare the solution code
            solution_list = list()

            # Example Solutions
            for solution in instance['solutions'][self.lang]:
                solution_list.append(solution)

            for solution in solution_list:
                result = self.venus_evaluation(solution, instance, timeout=90)
                print(result)


if __name__ == "__main__":
    venus_evaluator = VenusEvaluator(lang="python3")
    venus_evaluator.venus_pipeline()