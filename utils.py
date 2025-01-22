# code: utf-8

# Author: Du Mingzhe (mingzhe@nus.edu.sg)
# Date: 2025-01-13

import sys
sys.path.append('/home/nus_cisco_wp1/Projects/llm-sandbox')

import threading
from typing import List

def solution_filter(instance):
    filtered_solutions = list()
    solution_prompt = instance['code_prompt']
    
    # Runtime Solutions
    for solution in instance['rt_list']:
        if 'user.out' in solution['code']: continue
        if 'Solution' not in solution['code']: continue
        if solution_prompt not in solution['code']: continue
        filtered_solutions.append(solution)
    instance['rt_list'] = filtered_solutions
    
    # Memory Solutions
    filtered_solutions = list()
    for solution in instance['mm_list']:
        if 'user.out' in solution['code']: continue
        if 'Solution' not in solution['code']: continue
        if solution_prompt not in solution['code']: continue
        filtered_solutions.append(solution)
    instance['mm_list'] = filtered_solutions
    
    return instance


def run_with_timeout(func, timeout, *args, **kwargs):
    """Runs a function with a timeout."""
    result = {'output': None, 'error': None}

    def target():
        try:
            result['output'] = func(*args, **kwargs)
        except Exception as e:
            result['error'] = e

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        thread.join(0)  # Clean up the thread
        result["error"] = 'Timeout Reached.'
    return result

def solution_prompt_construct(instance: dict, instruction: str, solution_code: str) -> str:
    prompt = f"""
Given the following problem details and the original solution, outline your approach and then optimize the existing solution.

# Instruction:
{instruction}

# Problem Description:
{instance['content']}

# Original Solution:
{solution_code}

Please provide your response in **JSON format** with the following keys:
- "draft" : A concise explanation of your thought process or approach.
- "code"  : The optimized code solution.
    """
    return prompt

def code_evalution_construct(import_code: str, solution: str, test_case_generator: str, entry_point: str, test_case: List[dict]) -> str:
    test_case_template = """
test_case_input_obj = test_case_generator.decode_input('''{test_case_input_str}''')
test_case_output_obj = solution.{entry_point}(**test_case_input_obj)
test_case_output_str = test_case_generator.encode_output(test_case_output_obj)
assert test_case_output_str == '''{test_case_output_str}'''
"""
    test_code = ""
    for test in test_case:
        test_code += test_case_template.format(test_case_input_str=test['input'], test_case_output_str=test['output'], entry_point=entry_point)

    code_template = """
import json
import itertools
import collections
import heapq
import bisect
import string
import sys
import functools
import math
import copy
import re
# import numpy as np
# import pandas as pd

from math import floor, ceil, factorial, sqrt, inf
from sys import maxsize, stdin
from bisect import bisect_left, bisect_right
from itertools import permutations, zip_longest
from heapq import heappush, heappop, heapify
from collections import deque, defaultdict, OrderedDict, Counter
from typing import List, Optional, Tuple
from functools import lru_cache, cache

class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

### IMPORT CODE STARTS###
{import_code}
### IMPORT CODE ENDS###

### SOLUTION STARTS###
{solution}
### SOLUTION CODE ENDS###

### TEST CASE GENERATOR STARTS###
{test_case_generator}
### TEST CASE GENERATOR ENDS###

solution = Solution()
test_case_generator = TestCaseGenerator()

### TEST CASES STARTS###
{test_code}
### TEST CASES ENDS###

print('success')
    """
    return code_template.format(import_code=import_code, solution=solution, test_case_generator=test_case_generator, entry_point=entry_point, test_code=test_code)