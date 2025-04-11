# coding: utf-8

# Author: Du Mingzhe (mingzhe@nus.edu.sg)
# Date: 2025-04-10

import sys
sys.path.append("/home/mingzhe/Projects/Afterburner")

import time
import json
import utils
import requests
import textwrap
from tqdm import tqdm
from typing import Any, List
from tabulate import tabulate
from datasets import load_dataset, Dataset
from multiprocessing.dummy import Pool as ThreadPool

GENERATION_TEMPLATE = """
## Instructions
You are an expert competitive programmer who excels at solving algorithm problems in multiple programming languages.
Your task is to implement a solution to the following problem in {target_lang}.

## Problem Description
{question}

## Output Format
- Provide the complete solution code in **one markdown code block** with appropriate language identifier.
- Handle the input and output as specified in the problem statement.
- EXCLUDE ALL explanations and code comments.
"""

EVALUATION_TEMPLATE = """import io
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
        self.assertEqual(expected.strip(), actual.strip())
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

class TacoEvaluator:
    def __init__(self, lang, monolith_timeout, case_multiply, number_of_workers):
        self.lang = lang
        self.monolith_timeout = monolith_timeout
        self.case_multiply = case_multiply
        self.number_of_workers = number_of_workers

    @classmethod
    def taco_evaluation(cls, solution_code: str, test_case_list_str, case_multiply: int, timeout: int) -> dict:
        response = {'passed': False, 'time': float('inf'), 'memory': float('inf'), 'integral': float('inf'), 'status': 'error'}
        try:
            # Construct Test Code
            solution_code = textwrap.indent(solution_code, "\t")
            test_code = EVALUATION_TEMPLATE.format(solution_code=solution_code, test_case_list=test_case_list_str, case_multiply=case_multiply)

            with open("test_code.py", "w") as f:
                f.write(test_code)
            
            # Submit Test Code to Monolith
            data = {
                'code': test_code,
                'language': 'python',
                'libraries': [],
                'timeout': timeout,
                'run_profiling': True
            }
            monolith_response = requests.post(f'https://monolith.cool/execute', json=data, timeout=(120, timeout))
            if monolith_response.status_code == 200:
                monolith_response = monolith_response.json()

                response['status'] = monolith_response['status']
                if monolith_response["status"] == "success":
                    response['passed'] = True if monolith_response['output_dict']['stdout'] == 'Success\n' else False
                    response['time'] = monolith_response['output_dict']['duration']
                    response['memory'] = monolith_response['output_dict']['peak_memory']
                    response['integral'] = monolith_response['output_dict']['integral']
            elif monolith_response.status_code == 413:
                response['status'] = "too large"
            else:
                raise requests.exceptions.RequestException("API Error: " + str(monolith_response.content), monolith_response.status_code)
        except requests.exceptions.ReadTimeout as e:
            response['status'] = 'timeout (server)'
        except requests.exceptions.ConnectionError as e:
            response['status'] = 'timeout (client)'
        except Exception as e:
            print("Evaluation Error: ", e)
            response['status'] = 'error'
        finally:
            return response
    
    @classmethod
    def taco_generation(cls, inference_provider, model_name, instance: Any, target_lang: str, temperature=0, max_token=4096) -> dict:
        # Prepare the prompt
        prompt = GENERATION_TEMPLATE.format(
            target_lang=target_lang,
            question=instance['problem_content'],
            starter_code=utils.wrap_code_block(target_lang, instance['code_prompt']),
        )

        model_response = utils.code_generation(
            inference_provider=inference_provider,
            model_name=model_name,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_token
        )

        return model_response
    
    def taco_evalation_pipeline(self, data_precentage="100%"):
        ds = load_dataset("Elfsong/TACO_Python", split=f"train[:{data_precentage}]")

        solution_code_list = list()

        for instance in tqdm(ds):
            solutions = json.loads(instance['solutions'])
            input_output = json.loads(instance['input_output'])
            test_case_list_str = json.dumps([{"input": input_, "output": output_} for input_, output_ in zip(input_output['inputs'], input_output['outputs'])])
            
            for solution_code in solutions:
                solution_code=textwrap.indent(solution_code.strip(), "\t")
                solution_code_list.append((solution_code, test_case_list_str, self.case_multiply, self.monolith_timeout))
        print("Total solutions: ", len(solution_code_list))
        
        # Evaluate the solutions in parallel
        results = list()
        with ThreadPool(self.number_of_workers) as pool:
            with tqdm(total=len(solution_code_list), desc='Solution Evaluation') as pbar:
                for result in pool.imap(lambda args: TacoEvaluator.taco_evaluation(*args), solution_code_list):
                    results.append(result)
                    pbar.update(1)
        
        tc, pc = 0, 0
        for result in results:
            tc += 1
            if result['passed']:
                pc += 1
        print("Total test cases: ", tc)
        print("Passed test cases: ", pc)
        print("Pass rate: ", pc / tc)

if __name__ == "__main__":
    evaluator = TacoEvaluator(lang='python', monolith_timeout=90, case_multiply=100, number_of_workers=81)
    evaluator.taco_evalation_pipeline(data_precentage="50")


