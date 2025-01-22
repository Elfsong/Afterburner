# code: utf-8

# Author: Du Mingzhe (mingzhe@nus.edu.sg)
# Date: 2025-01-12

import sys
sys.path.append('/home/nus_cisco_wp1/Projects/llm-sandbox')

import time
import random
import threading
from tqdm import tqdm
from typing import List
from client import Client
from datasets import Dataset
from datasets import load_dataset
from llm_sandbox import SandboxSession
from pydantic import BaseModel, ConfigDict

class TestCaseGeneratorResponse(BaseModel):
    model_config = ConfigDict(extra='forbid')  # required for openai
    entry_point: str
    libraries: List[str]
    test_case_generator: str

class SetupCodeResponse(BaseModel):
    model_config = ConfigDict(extra='forbid')  # required for openai
    libraries: List[str]
    import_code: str

raw_ds = load_dataset("Elfsong/Venus", "python3", split="train")
new_ds = load_dataset("Elfsong/Venus_t", "python3", split="train")
question_id_list = [instance['question_id'] for instance in new_ds if instance['test_case']]
print('Current question:', len(question_id_list))

model = Client(model_name="gpt-4o", model_type="openai").model

def setup_prompt_construction(solution: str) -> str:
    prompt = """
    Identify all libraries required to execute the provided solution and generate the corresponding import statements.
    Return your response in JSON format, including the "libraries" list and "import_code".

    Solution:
    {solution}
    """
    return prompt.format(solution=solution)

def tc_prompt_construction(content: str, solution_prompt: str) -> str:
    prompt = """
    Based on the following Problem Description and the Solution Prompt, complete the implementation of the TestCaseGenerator class and identify the entry_point function in the solution.
    Generate test cases within a reasonable range; it is not necessary to cover the entire range stated in the problem description.
    Respond the 'TestCaseGenerator' and 'entry_point' in JSON format only.

    * Problem Description:
    {content}

    * Solutions Prompt:
    {solution_prompt}

    * TestCaseGenerator Template:
    class TestCaseGenerator:
        def generate(self) -> dict:
            # Generate a random test case input to be used for invoking the entry_point function in the solution prompt.
            pass

        def encode_input(self, input_obj) -> str:
            # Convert a test case input into a string
            pass

        def encode_output(self, output_obj) -> str:
            # Convert a test case output into a string
            pass

        def decode_input(self, input_str) -> dict:
            # Convert a test case input string into a Python dict
            pass
            
    * Example Usage:
    if __name__ == "__main__":
        solution = Solution()
        test_case_generator = TestCaseGenerator()

        # Generate and encode test case input
        test_case_input_obj = test_case_generator.generate()
        test_case_input_str = test_case_generator.encode_input(test_case_input_obj)

        # Decode input for validation
        test_case_input_obj = test_case_generator.decode_input(test_case_input_str)

        # Detect entry_point and compute output
        test_case_output_obj = solution.entry_point(**test_case_input_obj)
        test_case_output_str = test_case_generator.encode_output(test_case_output_obj)
    """
    return prompt.format(content=content, solution_prompt=solution_prompt)

def code_construction(solution: str, import_code: str, test_case_generator: str, entry_point: str) -> str:
    code = """
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

from math import floor, ceil, factorial, sqrt, inf
from sys import maxsize, stdin
from bisect import bisect_left, bisect_right
from itertools import permutations, zip_longest
from heapq import heappush, heappop, heapify
from collections import deque, defaultdict, OrderedDict
from typing import List, Optional, Tuple
from functools import lru_cache, cache

### IMPORT CODE STARTS###
{import_code}
### IMPORT CODE ENDS###

### SOLUTION STARTS###
{solution}
### SOLUTION CODE ENDS###

### TEST CASE GENERATOR STARTS###
{test_case_generator}
### TEST CASE GENERATOR ENDS###

if __name__ == "__main__":
    solution = Solution()
    test_case_generator = TestCaseGenerator()

    with open("test_cases", "w+") as f:
        for i in range(128):
            # Generate and encode test case input
            test_case_input_obj = test_case_generator.generate()
            test_case_input_str = test_case_generator.encode_input(test_case_input_obj)

            # Decode input for validation
            test_case_input_obj = test_case_generator.decode_input(test_case_input_str)

            # Detect entry_point and compute output
            test_case_output_obj = solution.{entry_point}(**test_case_input_obj)
            test_case_output_str = test_case_generator.encode_output(test_case_output_obj)
            
            test_case = {{"input": test_case_input_str, "output": test_case_output_str}}
            
            f.write(json.dumps(test_case)+'\\n')
    print("success")
    """
    return code.format(solution=solution, import_code=import_code, test_case_generator=test_case_generator, entry_point=entry_point)

def run_with_timeout(func, timeout, *args, **kwargs):
    """Runs a function with a timeout."""
    result = {'output': None, 'error': None}

    def target():
        try:
            result['output'] = func(*args, **kwargs)
        except Exception as e:
            result['error'] = e

    thread = threading.Thread(target=target)
    thread.start()
    
    thread.join(timeout)
    # for ts in range(timeout+1):
    #     time.sleep(1)
    #     if not thread.is_alive():
    #         break  # If the thread finishes, exit the loop
    #     ts += 1

    if thread.is_alive():
        thread.join(0)  # Clean up the thread
        result["error"] = 'Timeout Reached.'
    return result

instance_list = new_ds.to_list()

for instance in tqdm(list(raw_ds)[1400:]):
    try:
        content = instance['content']
        question_id = instance['question_id']
        solution_prompt = instance['code_prompt']
        solutions = instance['rt_list'] + instance['mm_list']

        if question_id in question_id_list: continue
        
        solutions = [solution for solution in solutions if ('open(' not in solution['code']) and ('print' not in solution['code']) and ('Solution' in solution['code'])]
        solution = random.choice(solutions)['code']
        
        tc_prompt = tc_prompt_construction(content=content, solution_prompt=solution_prompt)
        tc_response = model.json_generate(json_schema=TestCaseGeneratorResponse, prompt=tc_prompt)
        instance['test_case_generator'] = tc_response.test_case_generator
        instance['entry_point'] = tc_response.entry_point
        
        setup_prompt = setup_prompt_construction(solution=solution + instance['test_case_generator'])
        setup_response = model.json_generate(json_schema=SetupCodeResponse, prompt=setup_prompt)
        instance['libraries'] = setup_response.libraries if setup_response.libraries else []
        instance['import_code'] = setup_response.import_code if setup_response.import_code else ""
        
        code = code_construction(
            solution=solution, import_code=instance['import_code'], 
            test_case_generator=instance['test_case_generator'], entry_point=instance['entry_point']
        )
        
        with SandboxSession(lang="python", verbose=False) as session:
            session.setup(libraries=instance['libraries'])
            response = run_with_timeout(session.run, 60, code=code, run_memory_profile=False)
            output = response['output']

            if output.stdout == "success\n":
                session.copy_from_runtime('test_cases', 'test_cases')
                with open('test_cases', "r") as f:
                    instance['test_case'] = f.read()

            print(f"stdout: {output.stdout}")
            if output.stderr:
                raise Exception(f"stderr: {output.stderr}") 
            print("=" * 60)
        
        instance_list.append(instance)
        ds = Dataset.from_list(instance_list)
        ds.push_to_hub("Elfsong/Venus_t", 'python3', split='train')


    except Exception as e:
        print(f'Error: {e}')


