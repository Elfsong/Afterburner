
import time
import json
import textwrap
from tqdm import tqdm
from monolith import monolith
from multiprocessing import Pool
from datasets import load_dataset
from collections import OrderedDict


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

monolith = monolith.Monolith(backend_url='https://monolith.cool')

def apps_construction(solution_code, test_cases):
    solution_code = textwrap.indent(solution_code.strip(), "\t")
    test_case_list_str = json.dumps(test_cases, indent=4)

    test_code = TEMPLATE.format(
        solution_code=solution_code,
        test_case_list=test_case_list_str
    )
    return test_code

def apps_eval(solution_code, test_cases, timeout=60):
    try:
        test_code = apps_construction(solution_code, test_cases)
        task_id = monolith.post_code_submit(lang="python", libs=[], code=test_code, timeout=timeout, profiling=False)["task_id"]

        for _ in range(timeout):
            time.sleep(1)
            result = monolith.get_code_result(task_id)
            if result["status"] != "processing":
                break

        passed = False
        if result["status"] == "done":
            if result['output_dict']['stdout'] == 'Success\n':
                passed = True
        return passed
    except Exception as e:
        print(f"Error: {e}")
        return False
    
def apps_batch_eval(solution_code_list, test_cases, timeout=30):
    task_id_dict = OrderedDict()

    # Post code to Monolith
    for solution_code in solution_code_list:
        test_code = apps_construction(solution_code, test_cases)
        task_id = monolith.post_code_submit(lang="python", libs=[], code=test_code, timeout=timeout, profiling=False)["task_id"]
        task_id_dict[task_id] = None

    # Retrieve results from Monolith
    for _ in range(timeout):
        time.sleep(1)
        for task_id in [key for key in task_id_dict.keys() if task_id_dict[key] is None]:
            result = monolith.get_code_result(task_id)
            if result["status"] != "processing":
                if result["status"] == "done" and result['output_dict']['stdout'] == 'Success\n':
                    task_id_dict[task_id] = True
                else:
                    task_id_dict[task_id] = False
        if all(task_id_dict.values()):
            break
    return list(task_id_dict.values())


if __name__ == "__main__":
    ds = load_dataset("Elfsong/APPS", split="test")

    total_instances = len(ds)
    correct_count = 0

    for instance in ds:
        problem_id = instance["problem_id"]
        solutions = json.loads(instance["solutions"])
        test_cases = json.loads(instance["test_cases"])
        starter_code = instance["starter_code"]

        cases = []
        for solution in solutions:
            solution_code = f"\n# Starter Code\n{starter_code}\n# Solution Code\n{solution}"
            cases.append((solution_code, test_cases, 60))

        with Pool(8) as pool:
            outcomes = list(pool.starmap(apps_eval, cases))

        print(outcomes)

        if all(outcomes):
            print(f'[PASSED] - [{problem_id}]')
            correct_count += 1
        else:
            print(f'[FAILED] - [{problem_id}]')

    print(f"{correct_count} out of {total_instances} instances passed")