# coding: utf-8

# Author: Du Mingzhe (mingzhe@nus.edu.sg)
# Date: 2025-03-30

import sys
sys.path.append("/home/mingzhe/Projects/Afterburner")

import os
import time
import json
import utils
import random
import logging
import textwrap
import requests
from tqdm import tqdm
from tabulate import tabulate
from typing import Any, List, final
from datasets import load_dataset, Dataset, get_dataset_config_names
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
Your task is to implement a {efficiency_instruction} solution to the following problem in {target_lang}.

## Problem Description
{question}

## Starter Code
{starter_code}

## Output Format
- Provide the complete solution code in **one markdown code block** with appropriate language identifier.
- Implement the function with the exact signature (name, parameters, etc.) specified in the starter code.
- EXCLUDE ALL explanations, code comments, import/package/library statements, additional classes or functions outside of the starter code scope, or starting code like `if __name__ == "__main__":` or `func main()` or `package main` or `using namespace std;`.

Your solution:
"""

AFTERBURNER_GENERATION_TEMPLATE = """
## Instructions
You are an expert competitive programmer who excels at solving algorithm problems in multiple programming languages.
Your task is to implement a solution to the following problem in {target_lang}.

## Problem Description
{problem_description}

## Original Solution
{original_solution}

## Original Performance
Passed: {original_passed} / Time: {original_time} / Memory: {original_memory} / Integral: {original_integral}

## Output Format
- Provide the complete solution code in **one markdown code block** with appropriate language identifier.
- Fix the original solution if it was not passed. Optimize the {efficiency_instruction} performance if the original solution was passed.
- EXCLUDE ALL explanations, code comments, import/package/library statements, additional classes or functions outside of the starter code scope, or starting code like `if __name__ == "__main__":` or `func main()` or `package main` or `using namespace std;`.
"""

class VenusEvaluator:
    def __init__(self, lang, number_of_workers, case_multiply, monolith_timeout) -> None:
        self.lang = lang
        self.case_multiply = case_multiply
        self.monolith_timeout = monolith_timeout
        self.number_of_workers = number_of_workers
    
    @classmethod
    def venus_sync_evaluation(cls, solution_code: str, instance: Any, case_multiply: int, timeout: int) -> dict:
        response = {'passed': False, 'time': float('inf'), 'memory': float('inf'), 'integral': float('inf'), 'status': 'error'}
        try:
            # Construct Test Code
            test_case_runners = instance['test_case_runners']
            solution_code = test_case_runners.replace('==Code Submission==', solution_code.strip())
            solution_code = textwrap.indent(solution_code.strip(), "    ")
            
            test_case_evaluator = instance['test_case_evaluator'].strip()
            test_case_evaluator = textwrap.indent(test_case_evaluator, "    ")
            
            test_cases = json.loads(instance['test_cases'])
            test_case_list_str = json.dumps(test_cases, indent=4)

            test_code = EVALUATION_TEMPLATE.format(solution_code=solution_code, test_case_evaluator=test_case_evaluator, test_case_list=test_case_list_str, case_multiply=case_multiply)

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
    def venus_generation(cls, inference_provider, model_name, instance: Any, efficiency_instruction: str, target_lang: str, temperature=0, max_token=4096) -> str:
        # Check the Efficiency Instruction
        if efficiency_instruction not in utils.EFFICIENCY_INSTRUCTIONS:
            raise ValueError(f"Invalid efficiency instruction: {efficiency_instruction}")
        
        # Prepare the prompt
        prompt = GENERATION_TEMPLATE.format(
            target_lang=target_lang,
            question=instance['question_content'],
            starter_code=utils.wrap_code_block(target_lang, instance['code_prompt']),
            efficiency_instruction=utils.EFFICIENCY_INSTRUCTIONS[efficiency_instruction]
        )

        model_response = utils.model_inference(
            inference_provider=inference_provider,
            model_name=model_name,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_token
        )
        
        try:
            code = utils.extract_code_blocks(model_response)[0]['code']
        except Exception as e:
            print(f"[-] No code blocks found. Will return the whole response.")
            code = model_response
        return code
    
    @classmethod
    def venus_afterburner_generation(cls, inference_provider, model_name, instance: Any, original_solution: str, original_passed: bool, original_time: float, original_memory: float, original_integral: float, efficiency_instruction: str,target_lang: str, temperature=0, max_token=4096) -> str:
        # Prepare the prompt
        prompt = AFTERBURNER_GENERATION_TEMPLATE.format(
            target_lang=target_lang,
            problem_description=instance['question_content'],
            original_solution=original_solution,
            original_passed=original_passed,
            original_time=original_time,
            original_memory=original_memory,
            original_integral=original_integral,
            efficiency_instruction=utils.EFFICIENCY_INSTRUCTIONS[efficiency_instruction]
        )

        model_response = utils.model_inference(
            inference_provider=inference_provider,
            model_name=model_name,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_token
        )
        
        try:
            code = utils.extract_code_blocks(model_response)[-1]['code']
        except Exception as e:
            print(f"[-] No code blocks found. Will return the whole response: {e}")
            code = model_response
        return code
    
    def venus_distribution_pipeline(self):
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

                # Solution Deduplication
                solution_list = list(set(solution_list))

                # Prepare Test Packs (add code_prompt to each solution)
                solution_list = random.sample(solution_list, min(500, len(solution_list))) # remove it later
                test_packs = [(solution, instance, self.case_multiply, self.monolith_timeout) for solution in solution_list]

                print(f'[+] Problem {problem_id} [{index}/{len(leetcode_dataset)}] in [{i}% - {(i+1)}%] - {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')

                # Parallel Evaluation
                results = list()
                with ThreadPool(self.number_of_workers) as pool:
                    with tqdm(total=len(test_packs), desc='Solution Evaluation') as pbar:
                        for result in pool.imap(lambda args: VenusEvaluator.venus_sync_evaluation(*args), test_packs):
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

    def venus_afterburner_evaluation_pipeline(self, afterburner_model_name, afterburner_split_name, original_dataset_split_name, efficiency_instruction, inference_provider, force_generation=False, data_precentage="100%", data_multiply=1, mode="G+E"):
        # Generation Config
        generation_config = f"{original_dataset_split_name}_{efficiency_instruction}_{afterburner_split_name}"

        # Check the Efficiency Instruction
        if efficiency_instruction not in utils.EFFICIENCY_INSTRUCTIONS:
            raise ValueError(f"Invalid efficiency instruction: {efficiency_instruction}")

        # Evaluation Config
        print(f"[+] Processing Model: {generation_config}")
        print(f"[+] Data_Multiply: {data_multiply}")
        print(f"[+] Mode: {mode}")
        print(f"[+] Efficiency_Instruction: {efficiency_instruction}")
        print(f"[+] Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")

        # Meta Dataset
        venus_dataset = load_dataset("Elfsong/Venus_Python", split=f"test[:{data_precentage}]")

        # Original Generation Dataset
        generation_dict = dict()
        dataset_configs = get_dataset_config_names("Elfsong/Venus_Python_Model_Evaluation")
        if original_dataset_split_name in dataset_configs and not force_generation:
            generation_dataset = load_dataset("Elfsong/Venus_Python_Model_Evaluation", original_dataset_split_name, split="train")
            for instance in generation_dataset:
                problem_id = int(instance['problem_id'])
                if problem_id not in generation_dict:
                    generation_dict[problem_id] = [instance]
                else:
                    generation_dict[problem_id] += [instance]
        else:
            print(f"[Force_generation: {force_generation}] Original dataset split name {original_dataset_split_name} not found in dataset_configs, will set the original code to empty string")

        # Generation (G)
        if mode in ["G", "G+E"]:
            test_packs = list()
            for instance in tqdm(venus_dataset, desc='Generating solutions'):
                try:
                    # Find matching solutions directly
                    if int(instance['problem_id']) in generation_dict:
                        original_solutions = generation_dict[int(instance['problem_id'])]
                        original_solution_code = original_solutions[0]['solution_code']
                        original_solution_passed = all(s['passed'] for s in original_solutions)
                        original_solution_time = sum(s['absolute_time'] for s in original_solutions) / len(original_solutions)
                        original_solution_memory = sum(s['absolute_memory'] for s in original_solutions) / len(original_solutions)
                        original_solution_integral = sum(s['absolute_integral'] for s in original_solutions) / len(original_solutions)
                    else:
                        original_solution_code = "<No original code provided, please generate the code instead>"
                        original_solution_passed = False
                        original_solution_time = 90
                        original_solution_memory = 1000000
                        original_solution_integral = 90 * 1000000

                    generated_solution = self.venus_afterburner_generation(
                        inference_provider, 
                        afterburner_model_name, 
                        instance, 
                        original_solution_code, 
                        original_solution_passed, 
                        original_solution_time, 
                        original_solution_memory, 
                        original_solution_integral,
                        efficiency_instruction,
                        self.lang, 
                        temperature=0, 
                        max_token=8192
                    )
                except Exception as e:
                    print(f"[-] Generation Error: {e}")
                    generated_solution = ""
                finally:
                    test_packs.append({"problem_id": int(instance['problem_id']), "solution": generated_solution})
            ds = Dataset.from_list(test_packs)
            ds.push_to_hub("Elfsong/Venus_Model_Evaluation", generation_config, private=True)

        # Evaluation (E)
        if mode in ["E", "G+E"]:
            code_generation_dataset = load_dataset("Elfsong/Venus_Model_Evaluation", generation_config, split="train")

            # Check Dataset Size
            if len(code_generation_dataset) != len(venus_dataset):
                raise ValueError(f"Dataset size mismatch: {len(code_generation_dataset)} != {len(venus_dataset)}")
            
            test_packs = [(code['solution'], instance, self.case_multiply, self.monolith_timeout) for code, instance in zip(code_generation_dataset, venus_dataset)]
            test_packs = test_packs * data_multiply
                
            results = list()
            with ThreadPool(self.number_of_workers) as pool:
                with tqdm(total=len(test_packs), desc='Solution Evaluation') as pbar:
                    for result in pool.imap(lambda args: VenusEvaluator.venus_sync_evaluation(*args), test_packs):
                        results.append(result)
                        pbar.update(1)

            # Score Calculation
            instance_list = list()
            for instance, test_pack, result in zip(venus_dataset.repeat(data_multiply), test_packs, results):
                time_distribution = [s['time'] for s in instance['solutions'] if s['passed']]
                memory_distribution = [s['memory'] for s in instance['solutions'] if s['passed']]
                integral_distribution = [s['integral'] for s in instance['solutions'] if s['passed']]

                status = {
                    "problem_id": int(instance['problem_id']),
                    "passed": bool(result['passed']),
                    "precentile_time": utils.percentage_position(result['time'], time_distribution),
                    "precentile_memory": utils.percentage_position(result['memory'], memory_distribution),
                    "precentile_integral": utils.percentage_position(result['integral'], integral_distribution),
                    "absolute_time": float(result['time']),
                    "absolute_memory": float(result['memory']),
                    "absolute_integral": float(result['integral']),
                    "solution_code": str(test_pack[0]),
                }

                instance_list.append(status)
            
            scores = {"total_c": 0, "pass_c": 0, "time_s": 0,"memory_s": 0, "integral_s": 0}
            for instance in instance_list:
                scores["total_c"] += 1
                if instance['passed']:
                    scores["pass_c"] += 1
                    scores["time_s"] += instance['precentile_time']
                    scores["memory_s"] += instance['precentile_memory']
                    scores["integral_s"] += instance['precentile_integral']
            
            scores["pass_score"] = scores["pass_c"] / scores["total_c"]
            scores["time_score"] = scores["time_s"] / scores["total_c"]
            scores["memory_score"] = scores["memory_s"] / scores["total_c"]
            scores["integral_score"] = scores["integral_s"] / scores["total_c"]

            print(f"[+] Venus Evaluation")
            print(f"Generation Config: {generation_config}")
            print(f"Pass@1: {scores['pass_score']:.2f} Time_Precent: {scores['time_score']:.2f} Memory_Precent: {scores['memory_score']:.2f} Integral_Precent: {scores['integral_score']:.2f}")
                
            # Save the results
            ds = Dataset.from_list(instance_list)
            ds.push_to_hub("Elfsong/Venus_Python_Model_Evaluation", generation_config, private=True)

        print("========================================================")
        return generation_config

    def venus_evalution_pipeline(self, model_name, dataset_split_name, inference_provider, efficiency_instruction="integral", data_precentage="100%", data_multiply=1, mode="G+E"):
        dataset_split_name  = dataset_split_name + "_" + efficiency_instruction
        print(f"[+] Processing Model: {model_name} [{dataset_split_name}] in [{data_precentage}] - {data_multiply} - {mode} - {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")

        # Meta Dataset
        venus_dataset = load_dataset("Elfsong/Venus_Python", split=f"test[:{data_precentage}]")

        # Generation (G)
        if mode in ["G", "G+E"]:
            test_packs = list()
            for instance in tqdm(venus_dataset, desc='Generating solutions'):
                try:
                    generated_solution = self.venus_generation(inference_provider, model_name, instance, efficiency_instruction, self.lang, temperature=0, max_token=16384)
                except Exception as e:
                    print(f"[-] Generation Error: {e}")
                    generated_solution = ""
                finally:
                    test_packs.append({"problem_id": int(instance['problem_id']), "solution": generated_solution})
            ds = Dataset.from_list(test_packs)
            ds.push_to_hub("Elfsong/Venus_Model_Evaluation", dataset_split_name, private=True)
        
        # Parallel Evaluation (E)
        if mode in ["E", "G+E"]:
            code_generation_dataset = load_dataset("Elfsong/Venus_Model_Evaluation", dataset_split_name, split="train")
            
            solutions = dict()
            for instance in code_generation_dataset:
                solutions[int(instance['problem_id'])] = instance['solution']
                
            test_packs = [(solutions[int(instance['problem_id'])], instance, self.case_multiply, self.monolith_timeout) for instance in venus_dataset]
            test_packs = test_packs * data_multiply
                
            results = list()
            with ThreadPool(self.number_of_workers) as pool:
                with tqdm(total=len(test_packs), desc='Solution Evaluation') as pbar:
                    for result in pool.imap(lambda args: VenusEvaluator.venus_sync_evaluation(*args), test_packs):
                        results.append(result)
                        pbar.update(1)

            # Score Calculation
            instance_list = list()
            for instance, test_pack, result in zip(venus_dataset.repeat(data_multiply), test_packs, results):
                time_distribution = [s['time'] for s in instance['solutions'] if s['passed']]
                memory_distribution = [s['memory'] for s in instance['solutions'] if s['passed']]
                integral_distribution = [s['integral'] for s in instance['solutions'] if s['passed']]

                status = {
                    "problem_id": int(instance['problem_id']),
                    "passed": bool(result['passed']),
                    "precentile_time": utils.percentage_position(result['time'], time_distribution),
                    "precentile_memory": utils.percentage_position(result['memory'], memory_distribution),
                    "precentile_integral": utils.percentage_position(result['integral'], integral_distribution),
                    "absolute_time": float(result['time']),
                    "absolute_memory": float(result['memory']),
                    "absolute_integral": float(result['integral']),
                    "solution_code": str(test_pack[0]),
                }

                instance_list.append(status)
            
            scores = {"total_c": 0, "pass_c": 0, "time_s": 0,"memory_s": 0, "integral_s": 0}
            for instance in instance_list:
                scores["total_c"] += 1
                if instance['passed']:
                    scores["pass_c"] += 1
                    scores["time_s"] += instance['precentile_time']
                    scores["memory_s"] += instance['precentile_memory']
                    scores["integral_s"] += instance['precentile_integral']
            
            scores["pass_score"] = scores["pass_c"] / scores["total_c"]
            scores["time_score"] = scores["time_s"] / scores["total_c"]
            scores["memory_score"] = scores["memory_s"] / scores["total_c"]
            scores["integral_score"] = scores["integral_s"] / scores["total_c"]

            result = (f"Venus [{dataset_split_name}] Pass@1:{scores['pass_score']:.2f} Time_Precent:{scores['time_score']:.2f} Memory_Precent:{scores['memory_score']:.2f} Integral_Precent:{scores['integral_score']:.2f}")
            print(result)
                
            # Save the results
            ds = Dataset.from_list(instance_list)
            ds.push_to_hub("Elfsong/Venus_Python_Model_Evaluation", dataset_split_name, private=True)
        print("========================================================")

if __name__ == "__main__":
    venus_evaluator = VenusEvaluator(lang="python3", number_of_workers=81, case_multiply=64, monolith_timeout=90)

    # Get the distribution of each problem (it takes a long long time, be careful if you truely want to run it)
    # venus_evaluator.venus_distribution_pipeline()

    # Vanilla Evaluation
    # venus_evaluator.venus_evalution_pipeline(model_name="google/gemma-3-27b-it", dataset_split_name="gemma_3_27b", inference_provider="nebius", data_precentage="100%", data_multiply=16, mode="E", efficiency_instruction="time")
    # venus_evaluator.venus_evalution_pipeline(model_name="meta-llama/Llama-3.3-70B-Instruct", dataset_split_name="llama_3_3_70b_instruct", inference_provider="together", data_precentage="100%", data_multiply=16, mode="G", efficiency_instruction="time")
    # venus_evaluator.venus_evalution_pipeline(model_name="meta-llama/Llama-3.1-8B-Instruct", dataset_split_name="llama_3_1_8b_instruct", inference_provider="nebius", data_precentage="100%", data_multiply=16, mode="E", efficiency_instruction="time")
    # venus_evaluator.venus_evalution_pipeline(model_name="meta-llama/Llama-3.1-405B-Instruct", dataset_split_name="llama_3_1_405b_instruct", inference_provider="nebius", data_precentage="100%", data_multiply=16, mode="G", efficiency_instruction="time")
    # venus_evaluator.venus_evalution_pipeline(model_name="Qwen/Qwen2.5-Coder-32B-Instruct", dataset_split_name="qwen_2_5_coder_32b_instruct", inference_provider="nebius", data_precentage="100%", data_multiply=16, mode="G", efficiency_instruction="time")
    # venus_evaluator.venus_evalution_pipeline(model_name="Qwen/Qwen2.5-Coder-7B-Instruct", dataset_split_name="qwen_2_5_coder_7b_instruct", inference_provider="nebius", data_precentage="100%", data_multiply=16, mode="E", efficiency_instruction="time")
    # venus_evaluator.venus_evalution_pipeline(model_name="meta-llama/Llama-4-Scout-17B-16E-Instruct", dataset_split_name="llama_4_scout_17b_16e_instruct", inference_provider="together", data_precentage="100%", data_multiply=16, mode="E", efficiency_instruction="time")
    # venus_evaluator.venus_evalution_pipeline(model_name="deepseek-ai/DeepSeek-V3", dataset_split_name="deepseek_v3", inference_provider="together", data_precentage="100%", data_multiply=16, mode="E", efficiency_instruction="time")
    # venus_evaluator.venus_evalution_pipeline(model_name="microsoft/Phi-3-mini-4k-instruct", dataset_split_name="phi_3_mini_4k_instruct", inference_provider="nebius", data_precentage="100%", data_multiply=16, mode="E", efficiency_instruction="time")
    # venus_evaluator.venus_evalution_pipeline(model_name="gpt-4o", dataset_split_name="gpt_4o", inference_provider="openai", data_precentage="100%", data_multiply=16, mode="G+E", efficiency_instruction="time")
    # venus_evaluator.venus_evalution_pipeline(model_name="claude-3-7-sonnet-20250219", dataset_split_name="claude_3_7_sonnet", inference_provider="claude", data_precentage="100%", data_multiply=16, mode="E", efficiency_instruction="time")
    # venus_evaluator.venus_evalution_pipeline(model_name="claude-3-5-haiku-20241022", dataset_split_name="claude_3_5_haiku", inference_provider="claude", data_precentage="100%", data_multiply=16, mode="E", efficiency_instruction="time")
    # venus_evaluator.venus_evalution_pipeline(model_name="o3-mini", dataset_split_name="o3_mini", inference_provider="openai", data_precentage="100%", data_multiply=16, mode="E", efficiency_instruction="time")
    # venus_evaluator.venus_evalution_pipeline(model_name="Qwen/QwQ-32B", dataset_split_name="qwq_32b", inference_provider="nebius", data_precentage="100%", data_multiply=16, mode="E", efficiency_instruction="time")
    # venus_evaluator.venus_evalution_pipeline(model_name="Qwen/Qwen2.5-3B", dataset_split_name="qwen_2_5_3b", inference_provider="local", data_precentage="100%", data_multiply=16, mode="E", efficiency_instruction="time")
    # venus_evaluator.venus_evalution_pipeline(model_name="Qwen/Qwen2.5-3B", dataset_split_name="qwen_2_5_3b", inference_provider="local", data_precentage="100%", data_multiply=16, mode="G+E", efficiency_instruction="none")
    # venus_evaluator.venus_evalution_pipeline(model_name="/home/mingzhe/Projects/Afterburner/training/qwen_3b_sft_dpo_batch_2_ga_8_lr_4e-5/checkpoint-1200", dataset_split_name="qwen_2_5_3b_sft_dpo", inference_provider="local", data_precentage="100%", data_multiply=16, mode="E", efficiency_instruction="time")
    # venus_evaluator.venus_evalution_pipeline(model_name="Qwen/Qwen2.5-7B-Instruct", dataset_split_name="qwen_2_5_7b_instruct", inference_provider="local", data_precentage="100%", data_multiply=16, mode="G+E", efficiency_instruction="none")
    # venus_evaluator.venus_evalution_pipeline(model_name="Qwen/Qwen2.5-7B", dataset_split_name="qwen_2_5_7b", inference_provider="local", data_precentage="100%", data_multiply=16, mode="G", efficiency_instruction="time")
    # venus_evaluator.venus_evalution_pipeline(model_name="Qwen/Qwen2.5-Coder-7B", dataset_split_name="qwen_2_5_coder_7b", inference_provider="local", data_precentage="100%", data_multiply=16, mode="E", efficiency_instruction="time")
    # venus_evaluator.venus_evalution_pipeline(model_name="o4-mini", dataset_split_name="o4_mini", inference_provider="openai", data_precentage="100%", data_multiply=16, mode="E", efficiency_instruction="time")
    # venus_evaluator.venus_evalution_pipeline(model_name="gemini-2.5-pro-preview-03-2", dataset_split_name="gemini_2_5_pro", inference_provider="gemini", data_precentage="5", data_multiply=16, mode="G", efficiency_instruction="time")
    # venus_evaluator.venus_evalution_pipeline(model_name="Qwen/", dataset_split_name="gemini_2_5_pro", inference_provider="gemini", data_precentage="5", data_multiply=16, mode="G", efficiency_instruction="time")

    
    # Afterburner Evaluation
    # venus_evaluator.venus_afterburner_evaluation_pipeline(afterburner_model_name="gpt-4o", afterburner_split_name="gpt_4o", original_dataset_split_name="qwen_2_5_coder_7b", efficiency_instruction="integral", inference_provider="openai", data_precentage="100%", data_multiply=16, mode="G+E")
    # venus_evaluator.venus_afterburner_evaluation_pipeline(afterburner_model_name="gpt-4o", afterburner_split_name="gpt_4o", original_dataset_split_name="qwen_2_5_3b", efficiency_instruction="integral", inference_provider="openai", data_precentage="100%", data_multiply=16, mode="G+E")
    # venus_evaluator.venus_afterburner_evaluation_pipeline(afterburner_model_name="gpt-4o", afterburner_split_name="gpt_4o", original_dataset_split_name="qwen_2_5_7b_instruct", efficiency_instruction="integral", inference_provider="openai", data_precentage="100%", data_multiply=16, mode="E")
    # venus_evaluator.venus_afterburner_evaluation_pipeline(afterburner_model_name="gpt-4o", afterburner_split_name="gpt_4o", original_dataset_split_name="gpt_4o", efficiency_instruction="integral", inference_provider="openai", data_precentage="100%", data_multiply=16, mode="G+E")
    # venus_evaluator.venus_afterburner_evaluation_pipeline(afterburner_model_name="gpt-4o", afterburner_split_name="gpt_4o", original_dataset_split_name="llama_4_scout_17b_16e_instruct", efficiency_instruction="integral", inference_provider="openai", data_precentage="100%", data_multiply=16, mode="E")
    # venus_evaluator.venus_afterburner_evaluation_pipeline(afterburner_model_name="gpt-4o", afterburner_split_name="gpt_4o", original_dataset_split_name="qwen_2_5_3b", efficiency_instruction="time", inference_provider="openai", data_precentage="100%", data_multiply=16, mode="E")
    # venus_evaluator.venus_afterburner_evaluation_pipeline(afterburner_model_name="gpt-4o", afterburner_split_name="gpt_4o", original_dataset_split_name="qwen_2_5_3b", efficiency_instruction="memory", inference_provider="openai", data_precentage="100%", data_multiply=16, mode="E")
    # venus_evaluator.venus_afterburner_evaluation_pipeline(afterburner_model_name="gpt-4o", afterburner_split_name="gpt_4o", original_dataset_split_name="qwen_2_5_coder_7b", efficiency_instruction="time", inference_provider="openai", data_precentage="100%", data_multiply=16, mode="E")
    # venus_evaluator.venus_afterburner_evaluation_pipeline(afterburner_model_name="gpt-4o", afterburner_split_name="gpt_4o", original_dataset_split_name="qwen_2_5_coder_7b", efficiency_instruction="memory", inference_provider="openai", data_precentage="100%", data_multiply=16, mode="E")
    # venus_evaluator.venus_afterburner_evaluation_pipeline(afterburner_model_name="gpt-4o", afterburner_split_name="gpt_4o", original_dataset_split_name="qwen_2_5_7b_instruct", efficiency_instruction="time", inference_provider="openai", data_precentage="100%", data_multiply=16, mode="E")
    # venus_evaluator.venus_afterburner_evaluation_pipeline(afterburner_model_name="gpt-4o", afterburner_split_name="gpt_4o", original_dataset_split_name="qwen_2_5_7b_instruct", efficiency_instruction="memory", inference_provider="openai", data_precentage="100%", data_multiply=16, mode="E")
    # venus_evaluator.venus_afterburner_evaluation_pipeline(afterburner_model_name="gpt-4o", afterburner_split_name="gpt_4o", original_dataset_split_name="llama_4_scout_17b_16e_instruct", efficiency_instruction="time", inference_provider="openai", data_precentage="100%", data_multiply=16, mode="E")
    # venus_evaluator.venus_afterburner_evaluation_pipeline(afterburner_model_name="gpt-4o", afterburner_split_name="gpt_4o", original_dataset_split_name="llama_4_scout_17b_16e_instruct", efficiency_instruction="memory", inference_provider="openai", data_precentage="100%", data_multiply=16, mode="E")
    # venus_evaluator.venus_afterburner_evaluation_pipeline(afterburner_model_name="claude-3-7-sonnet-20250219", afterburner_split_name="claude_3_7_sonnet", original_dataset_split_name="gpt_4o", efficiency_instruction="integral", inference_provider="claude", data_precentage="100%", data_multiply=16, mode="E")
    # venus_evaluator.venus_afterburner_evaluation_pipeline(afterburner_model_name="claude-3-7-sonnet-20250219", afterburner_split_name="claude_3_7_sonnet", original_dataset_split_name="gpt_4o", efficiency_instruction="time", inference_provider="claude", data_precentage="100%", data_multiply=16, mode="E")
    # venus_evaluator.venus_afterburner_evaluation_pipeline(afterburner_model_name="claude-3-7-sonnet-20250219", afterburner_split_name="claude_3_7_sonnet", original_dataset_split_name="gpt_4o", efficiency_instruction="memory", inference_provider="claude", data_precentage="100%", data_multiply=16, mode="E")
    # venus_evaluator.venus_afterburner_evaluation_pipeline(afterburner_model_name="claude-3-7-sonnet-20250219", afterburner_split_name="claude_3_7_sonnet", original_dataset_split_name="claude_3_5_haiku", efficiency_instruction="integral", inference_provider="claude", data_precentage="100%", data_multiply=16, mode="E")
    # venus_evaluator.venus_afterburner_evaluation_pipeline(afterburner_model_name="claude-3-7-sonnet-20250219", afterburner_split_name="claude_3_7_sonnet", original_dataset_split_name="claude_3_5_haiku", efficiency_instruction="time", inference_provider="claude", data_precentage="100%", data_multiply=16, mode="E")
    # venus_evaluator.venus_afterburner_evaluation_pipeline(afterburner_model_name="claude-3-7-sonnet-20250219", afterburner_split_name="claude_3_7_sonnet", original_dataset_split_name="claude_3_5_haiku", efficiency_instruction="memory", inference_provider="claude", data_precentage="100%", data_multiply=16, mode="E")
    # venus_evaluator.venus_afterburner_evaluation_pipeline(afterburner_model_name="claude-3-7-sonnet-20250219", afterburner_split_name="claude_3_7_sonnet", original_dataset_split_name="claude_3_7_sonnet", efficiency_instruction="integral", inference_provider="claude", data_precentage="100%", data_multiply=16, mode="E")
    # venus_evaluator.venus_afterburner_evaluation_pipeline(afterburner_model_name="claude-3-7-sonnet-20250219", afterburner_split_name="claude_3_7_sonnet", original_dataset_split_name="claude_3_7_sonnet", efficiency_instruction="time", inference_provider="claude", data_precentage="100%", data_multiply=16, mode="E")
    # venus_evaluator.venus_afterburner_evaluation_pipeline(afterburner_model_name="claude-3-7-sonnet-20250219", afterburner_split_name="claude_3_7_sonnet", original_dataset_split_name="claude_3_7_sonnet", efficiency_instruction="memory", inference_provider="claude", data_precentage="100%", data_multiply=16, mode="E")
    # venus_evaluator.venus_afterburner_evaluation_pipeline(afterburner_model_name="claude-3-7-sonnet-20250219", afterburner_split_name="claude_3_7_sonnet", original_dataset_split_name="deepseek_v3", efficiency_instruction="integral", inference_provider="claude", data_precentage="100%", data_multiply=16, mode="E")
    # venus_evaluator.venus_afterburner_evaluation_pipeline(afterburner_model_name="claude-3-7-sonnet-20250219", afterburner_split_name="claude_3_7_sonnet", original_dataset_split_name="deepseek_v3", efficiency_instruction="time", inference_provider="claude", data_precentage="100%", data_multiply=16, mode="E")
    # venus_evaluator.venus_afterburner_evaluation_pipeline(afterburner_model_name="claude-3-7-sonnet-20250219", afterburner_split_name="claude_3_7_sonnet", original_dataset_split_name="deepseek_v3", efficiency_instruction="memory", inference_provider="claude", data_precentage="100%", data_multiply=16, mode="E")
    # venus_evaluator.venus_afterburner_evaluation_pipeline(afterburner_model_name="Qwen/QwQ-32B", afterburner_split_name="qwq_32b", original_dataset_split_name="claude_3_7_sonnet", efficiency_instruction="integral", inference_provider="nebius", data_precentage="100%", data_multiply=16, mode="G")
    # venus_evaluator.venus_afterburner_evaluation_pipeline(afterburner_model_name="Qwen/QwQ-32B", afterburner_split_name="qwq_32b", original_dataset_split_name="claude_3_7_sonnet", efficiency_instruction="time", inference_provider="nebius", data_precentage="100%", data_multiply=16, mode="G")
    # venus_evaluator.venus_afterburner_evaluation_pipeline(afterburner_model_name="Qwen/QwQ-32B", afterburner_split_name="qwq_32b", original_dataset_split_name="claude_3_7_sonnet", efficiency_instruction="memory", inference_provider="nebius", data_precentage="100%", data_multiply=16, mode="G")
    # venus_evaluator.venus_afterburner_evaluation_pipeline(afterburner_model_name="Elfsong/Afterburner_3B_120", afterburner_split_name="afterburner_120", original_dataset_split_name="qwen_2_5_3b_instruct", efficiency_instruction="time", inference_provider="local", data_precentage="100%", data_multiply=16, mode="G")
    # venus_evaluator.venus_afterburner_evaluation_pipeline(afterburner_model_name="Elfsong/Afterburner_3B_120", afterburner_split_name="afterburner_120", original_dataset_split_name="qwen_2_5_3b_instruct", efficiency_instruction="memory", inference_provider="local", data_precentage="100%", data_multiply=16, mode="G")
    # venus_evaluator.venus_afterburner_evaluation_pipeline(afterburner_model_name="Elfsong/Afterburner_3B_120", afterburner_split_name="afterburner_120", original_dataset_split_name="qwen_2_5_3b_instruct", efficiency_instruction="integral", inference_provider="local", data_precentage="100%", data_multiply=16, mode="G")
    # venus_evaluator.venus_afterburner_evaluation_pipeline(afterburner_model_name="Elfsong/Afterburner_3B_120", afterburner_split_name="afterburner_100", original_dataset_split_name="qwen_2_5_3b_instruct", efficiency_instruction="time", inference_provider="local", data_precentage="100%", data_multiply=16, mode="G")
    # venus_evaluator.venus_afterburner_evaluation_pipeline(afterburner_model_name="Elfsong/Afterburner_3B_120", afterburner_split_name="afterburner_100", original_dataset_split_name="qwen_2_5_3b_instruct", efficiency_instruction="memory", inference_provider="local", data_precentage="100%", data_multiply=16, mode="G")
    # venus_evaluator.venus_afterburner_evaluation_pipeline(afterburner_model_name="Elfsong/Afterburner_3B_120", afterburner_split_name="afterburner_100", original_dataset_split_name="qwen_2_5_3b_instruct", efficiency_instruction="integral", inference_provider="local", data_precentage="100%", data_multiply=16, mode="G")
    # venus_evaluator.venus_afterburner_evaluation_pipeline(afterburner_model_name="Elfsong/Afterburner_3B_120", afterburner_split_name="afterburner_120", original_dataset_split_name="qwen_2_5_7b_instruct", efficiency_instruction="time", inference_provider="local", data_precentage="100%", data_multiply=16, mode="E")
    # venus_evaluator.venus_afterburner_evaluation_pipeline(afterburner_model_name="Elfsong/Afterburner_3B_120", afterburner_split_name="afterburner_120", original_dataset_split_name="qwen_2_5_7b_instruct", efficiency_instruction="memory", inference_provider="local", data_precentage="100%", data_multiply=16, mode="G")
    # venus_evaluator.venus_afterburner_evaluation_pipeline(afterburner_model_name="Elfsong/Afterburner_3B_120", afterburner_split_name="afterburner_120", original_dataset_split_name="qwen_2_5_7b_instruct", efficiency_instruction="integral", inference_provider="local", data_precentage="100%", data_multiply=16, mode="G")
    
    # Iteration Evaluation
    efficiency_instruction = "time"
    dataset_split_name = "qwen_2_5_3b"
    for i in tqdm(range(8), desc=f"[{dataset_split_name}] [{efficiency_instruction}] Iteration Evaluation"):
        dataset_split_name = venus_evaluator.venus_afterburner_evaluation_pipeline(afterburner_model_name="Qwen/Qwen2.5-3B", afterburner_split_name="qwen_2_5_3b", original_dataset_split_name=dataset_split_name, efficiency_instruction=efficiency_instruction, force_generation=True, inference_provider="local", data_precentage="100%", data_multiply=4, mode="G")
    
