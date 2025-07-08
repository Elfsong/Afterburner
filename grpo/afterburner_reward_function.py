# coding=utf-8

# Author: Mingzhe Du (mingzhe@nus.edu.sg)
# Date: 2025-04-10

import re
import time
import json
import random
import debugpy
import requests
import textwrap
import itertools
import numpy as np
from tqdm import tqdm
from multiprocessing.pool import ThreadPool


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

def scale_clip(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, -1.0, 1.0)

def safe_delta(baseline: np.ndarray, current: np.ndarray, scale: float, clip: float) -> np.ndarray:
    baseline = np.clip(baseline, 0, clip)
    current = np.clip(current, 0, clip)
    gain = (baseline - current) / (baseline + 1e-9)
    gain = np.clip(gain, -1.0, 1.0)
    return np.tanh(scale * gain)   

def safe_minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = x.min(), x.max()
    normed =  (x - lo) / (hi - lo + 1e-8)
    return scale_clip(normed)

def zscore(x: np.ndarray) -> np.ndarray:
    mu, sigma = x.mean(), x.std()
    return (x - mu) / (sigma + 1e-9)

def pass_reward(baseline: np.ndarray, current: np.ndarray) -> np.ndarray:
    r = np.full(baseline.shape, -0.5, dtype=np.float32)      # maintain failed
    r[baseline.astype(bool) & current.astype(bool)]  = 0.5   # maintain passed
    r[~baseline.astype(bool) & current.astype(bool)] = 1.0   # new pass
    r[baseline.astype(bool) & ~current.astype(bool)] = -1.0  # degradation
    return r

def single_thinking_solution_format(text: str) -> bool:
    pattern = re.compile(
        r"""
        \A\s*                      # optional leading whitespace
        <thinking>                 # start <thinking>
            (?:(?!<thinking>).)*?  # allow any content, but no new <thinking>
        </thinking>\s*             # end <thinking>
        <solution>                 # start <solution>  
            (?:(?!<thinking>|<solution>).)*?  # allow any content, but no new <thinking> or <solution>    
        </solution>\s*             # end <solution>
        \Z                         # end of string
        """,
        re.DOTALL | re.VERBOSE,
    )
    return bool(pattern.fullmatch(text))

def repeated_thinking_solution_format(text: str) -> bool:
    pattern = r"""
        \A                      # start of string
        (?:                     # --- one or more of the following group ---
            \s*<think>.*?</think>\s*      # a <think> block (non‑greedy)
            <solution>.*?</solution>\s*   # immediately followed by <solution>
        )+                      # repeat the pair 1+ times
        \Z                      # end of string
    """
    return bool(re.fullmatch(pattern, text, flags=re.DOTALL | re.VERBOSE))

def extract_solution_blocks(text: str) -> list[str]:
    pattern = r"<solution>(.*?)</solution>"
    matches = re.findall(pattern, text, flags=re.DOTALL)
    return [match.strip() for match in matches] if matches else []

def extract_code_blocks(text: str) -> list[dict[str, str]]:
    _CODE_BLOCK_PATTERN = re.compile(r"```([\w+-]*)(?:\n|\r\n)?(.*?)```", re.DOTALL)
    blocks: list[dict[str, str]] = []
    for match in _CODE_BLOCK_PATTERN.finditer(text):
        lang = match.group(1).strip()
        code = match.group(2).strip()
        blocks.append({"lang": lang, "code": code})
    return blocks

def performance_evalution(solution_str: str, extra_info: dict) -> dict:
    # Integrate functional correctness and efficiency
    response = {'passed': False, 'time': 90000, 'memory': 1048576, 'integral': 1048576*90000, 'status': 'error'}
    try:
        # Extract the solution code from the solution string
        solution_code_str = ""
        solution_blocks = extract_solution_blocks(solution_str)
        if solution_blocks:
            solution_code_blocks = extract_code_blocks(solution_blocks[-1])
            if solution_code_blocks:
                solution_code_str = solution_code_blocks[-1]['code']
        
        # Construct Test Code
        instance = extra_info['instance']
        case_multiply = extra_info['case_multiply']
        test_case_runners = instance['test_case_runners']
        solution_code = test_case_runners.replace('==Code Submission==', solution_code_str.strip())
        solution_code = textwrap.indent(solution_code.strip(), "    ")
        
        test_case_evaluator = instance['test_case_evaluator'].strip()
        test_case_evaluator = textwrap.indent(test_case_evaluator, "    ")
        
        test_cases = json.loads(instance['test_cases'])
        test_case_list_str = json.dumps(test_cases, indent=4)

        test_code = EVALUATION_TEMPLATE.format(
            solution_code=solution_code, 
            test_case_evaluator=test_case_evaluator, 
            test_case_list=test_case_list_str, 
            case_multiply=case_multiply
        )

        # Submit Test Code to Monolith
        data = {
            'code': test_code,
            'language': 'python',
            'libraries': [],
            'timeout': 90,
            'run_profiling': True
        }
        monolith_response = requests.post(f'https://monolith.cool/execute', json=data, timeout=90)
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
        response['status'] = 'error'
    finally:
        return response

def improvement_reward_fn_batch(data_sources, solution_strs, ground_truths, extra_infos=None) -> list[float]:        
    # Send the batch request to Monolith
    
    monolith_responses = list()
    worker_num = min(len(data_sources), 81)
    progress_bar = tqdm(total=len(solution_strs), desc=f"Evaluating solutions with Monolith [workers: {worker_num}]")
    with ThreadPool(worker_num) as pool:
        for response in pool.imap(lambda args: performance_evalution(*args), zip(solution_strs, extra_infos)):
            if response['passed'] == False:
                response['time'] = 90
                response['memory'] = 1e9
                response['integral'] = 1e10
            monolith_responses.append(response)
            progress_bar.update(1)
            
    # Baseline scores    
    baseline_fc_scores = np.array([baseline['passed'] for baseline in ground_truths])
    baseline_time_scores = np.array([baseline['time'] for baseline in ground_truths])
    baseline_memory_scores = np.array([baseline['memory'] for baseline in ground_truths])
    baseline_integral_scores = np.array([baseline['integral'] for baseline in ground_truths])
    
    # Reward Components
    fc_scores = np.array([response['passed'] for response in monolith_responses])
    time_scores = np.array([response['time'] for response in monolith_responses])
    memory_scores = np.array([response['memory'] for response in monolith_responses])
    integral_scores = np.array([response['integral'] for response in monolith_responses])
    
    # Improvement scores
    improvement_scores  = pass_reward(baseline_fc_scores, fc_scores)
    time_scores         = safe_delta(baseline_time_scores, time_scores, scale=1.0, clip=90)
    memory_scores       = safe_delta(baseline_memory_scores, memory_scores, scale=1.0, clip=1048576)
    integral_scores     = safe_delta(baseline_integral_scores, integral_scores, scale=1.0, clip=1048576*90)
    
    # Calculate the final scores
    scores = list()
    for extra_info, fc_score, time_score, memory_score, integral_score in zip(extra_infos, improvement_scores, time_scores, memory_scores, integral_scores):
        efficiency_instruction = extra_info['efficiency_instruction']
        
        # Consider efficiency only when the solution is passed
        efficiency_weight = 0.5 if fc_score > 0 else 0 
        
        if efficiency_instruction == 'integral':
            scores.append(efficiency_weight * integral_score + fc_score)
        elif efficiency_instruction == 'time':
            scores.append(efficiency_weight * time_score + fc_score)
        elif efficiency_instruction == 'memory':
            scores.append(efficiency_weight * memory_score + fc_score)
    return scores

def format_reward_fn_batch(data_sources, solution_strs, ground_truths, extra_infos=None) -> list[float]:
    scores = list()
    for solution_str in solution_strs:
        if single_thinking_solution_format(solution_str):
            scores.append(1)
        else:
            scores.append(0)

    return scores

def length_reward_fn_batch(data_sources, solution_strs, ground_truths, extra_infos=None) -> list[float]:
    raw_scores = [len(solution_str) for solution_str in solution_strs]
    scores = safe_minmax(raw_scores)
    return scores

def afterburner_reward_fn_batch(data_sources, solution_strs, ground_truths, extra_infos=None) -> list[float]:
    print(f"[+]  Afterburner Reward Function Batch [Solution Size: {len(solution_strs)}]")
    start_time = time.time()
    
    # Debug Hook Point
    # debugpy.listen(("0.0.0.0", 5678))
    # print("[🧐] Waiting for Debugger Attach...")
    # debugpy.wait_for_client()
            
    # # Improvement Reward
    improvement_scores = improvement_reward_fn_batch(data_sources, solution_strs, ground_truths, extra_infos)
    
    # # Format Reward
    format_scores = format_reward_fn_batch(data_sources, solution_strs, ground_truths, extra_infos)
    
    # # Length Reward
    # length_scores = length_reward_fn_batch(data_sources, solution_strs, ground_truths, extra_infos)
    
    # Reward Component Combination
    beta_l = 0.3     # weight for length_score
    beta_i = 0.5     # weight for improvement_score
    beta_f = 0.2     # weight for format_score

    afterburner_scores = list()
    for length_score, improvement_score, format_score in zip(itertools.repeat(0), improvement_scores, format_scores):
        # Additive Combination
        score = (beta_l * length_score) + (beta_i * improvement_score) + (beta_f * format_score)
        print(f"[-] improvement: {improvement_score:.2f} \t\t format: {format_score:.2f} \t\t final: {score:.2f}")
        afterburner_scores.append(score)
    
    end_time = time.time()
    print(f"[+]  Afterburner Reward Function Batch [Time Cost: {end_time - start_time:.2f}s]")

    return afterburner_scores
