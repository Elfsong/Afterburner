# coding: utf-8

# Just-In-Time Tester (JITT)
# Author: Du Mingzhe (mingzhe@nus.edu.sg)
# Date: 2025-03-15

import re
import os
import json
import time
import random
import logging
import requests
from tqdm import tqdm
from typing import List
from openai import OpenAI


PYTHON_TEST_CASE_GENERATOR_TEMPLATE = """
Please develop a structured Python Class TestCaseGenerator for a given Python Solution, containing the following components:

Python Solution:
```python
{python_solution}
```

1. Input Generator
    def generate_test_input() -> TestInput:
    '''Generate random test input objects to comprehensively test the Python solution'''
    
2. Output Generator
    def generate_expected_output(input: TestInput) -> TestOutput:
    '''Generates expected output by executing the Python solution with given input'''
    
3. Input Serialization
    def serialize_input(obj: TestInput) -> str:
    '''Serializes input object to string'''
    
4. Input Deserialization
    def deserialize_input(s: str) -> TestInput:
    '''Restores input object from serialized string'''
    
5. Output Serialization
    def serialize_output(obj: TestOutput) -> str:
    '''Serializes output object to string'''
    
6. Output Deserialization
    def deserialize_output(s: str) -> TestOutput:
    '''Restores output object from serialized string'''

Additional Requirements:

- Use type annotations.
- Maintain pure logic functions (avoid side effects).
- Call the Python solution to generate expected output. Don't modify the solution.
- Assuming the Python Solution class is already defined in the same file, respond the executable TestCaseGenerator ONLY.
"""

PYTHON_TEST_CASE_CONSTRUCTION_TEMPLATE = """
Analyze the provided code, identify any missing required libraries, and generate a response containing:
- A list of Python libraries that need to be installed (excluding standard library modules)
- A refined version of the code with all necessary import statements added at the beginning

Format your response as valid JSON with these keys:
- "libraries": Array of strings (library names)
- "import_statements": String containing all necessary import statements
- "executable_code": String containing complete working code with proper imports

Ensure:
- Import statements are organized following PEP8 guidelines.
- Only include libraries actually used in the code.
- Maintain the original code's functionality.
- Use double quotes for JSON formatting.
- Keep # <solution_code> and # </solution_code> tags to encapsulate the python_solution_code.

# <solution_code>
{python_solution_code}
# </solution_code>

# <test_case_generator_code>
{test_case_generator_code}
# </test_case_generator_code>

# Test Case Construction
test_case_generator = TestCaseGenerator()

cases = list()
for _ in range({number_of_cases}):
    try:
        test_input = test_case_generator.generate_test_input()
        test_output = test_case_generator.generate_expected_output(test_input)
        
        test_input_str = test_case_generator.serialize_input(test_input)
        test_input_restored = test_case_generator.deserialize_input(test_input_str)
        
        test_output_str = test_case_generator.serialize_output(test_output)
        test_output_restored = test_case_generator.deserialize_output(test_output_str)
        
        test_output = test_case_generator.generate_expected_output(test_input_restored)
        if test_output == test_output_restored:
            cases.append({{"input": test_input, "output": test_output}})
    except Exception as e:
        pass
        
print("<case_data>")
print(json.dumps(cases))
print("</case_data>")
"""

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('jitt.log'),
        logging.StreamHandler()
    ]
)

class JITT:
    def __init__(self, number_of_cases=1, timeout=60):
        self.remote_sandbox = 'https://monolith.cool'
        self.number_of_cases = number_of_cases
        self.timeout = timeout
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.client = OpenAI(api_key=self.openai_api_key)
        self.logger = logging.getLogger('jitt_logger')
        
        self.python_test_case_generator_template = PYTHON_TEST_CASE_GENERATOR_TEMPLATE
        self.python_test_case_construction_template = PYTHON_TEST_CASE_CONSTRUCTION_TEMPLATE
    
    def openai_call(self, prompt: str) -> str:
        completion = self.client.chat.completions.create(
            model="o3-mini",
            messages=[
                {"role": "system",   "content": "You're a world-class programmer."},
                {"role": "user",     "content": prompt},
            ]
        )
        return completion.choices[0].message.content
        
    def get_test_case_generator(self, python_solution: str) -> str:
        prompt = self.python_test_case_generator_template.format(
            python_solution=python_solution
        )
        response = self.openai_call(prompt)
        return response
        
    
    def get_test_case_construction(self, python_solution_code, test_case_generator_code) -> str:
        prompt = self.python_test_case_construction_template.format(
            python_solution_code=python_solution_code, 
            test_case_generator_code=test_case_generator_code, 
            number_of_cases=self.number_of_cases
        )
        
        response = self.openai_call(prompt)
        try:
            response = json.loads(response)
        except Exception as e:
            self.logger.error(f'Test Case Construction Code Generation Error: {e}')
            response = None
        return response

    def post_code_submit(self, libs: List, code: str, timeout: int, profiling: bool) -> str:
        data = {
            'language': "python",
            'code': code,
            'libraries': libs,
            'timeout': timeout,
            'run_memory_profile': profiling
        }

        response = requests.post('https://monolith.cool/execute', json=data)
        task_id = response.json()['task_id']
        return task_id
    
    def get_code_result(self, task_id: str) -> str:
        response = requests.get(f'https://monolith.cool/results/{task_id}')
        return response.json()
    
    def test_case_construction(self, response: str) -> str:
        try:
            test_cases_json_str = "[]"
            match = re.search(r"<case_data>(.*?)</case_data>", response, re.DOTALL)
            if match:
                test_cases_json_str = match.group(1)
            test_cases_json = json.loads(test_cases_json_str)
            return test_cases_json
        except Exception as e:
            self.logger.error(f'Test Case Construction Error: {e}')
            return []
    
    def generate(self, solution_code: str) -> List:
        try:
            self.logger.info(f'Getting Test Case Generator')
            test_case_generator_code = self.get_test_case_generator(solution_code)
            
            self.logger.info(f'Getting Test Case Construction')
            test_case_construction = self.get_test_case_construction(solution_code, test_case_generator_code)
            
            self.logger.info(f'Parsing Test Case Construction Code')
            if not test_case_construction: return []
            
            self.logger.info(f'Submitting code to Monolith')
            task_id = self.post_code_submit(
                test_case_construction['libraries'], 
                test_case_construction['executable_code'], 
                timeout=self.timeout, 
                profiling=False
            )
            
            self.logger.info(f'Retrieving Test Cases')
            response = {'status': 'pending'}
            for _ in tqdm(range(self.timeout), desc='Waiting for test case generation...'):
                time.sleep(1)
                response = self.get_code_result(task_id)
                if response['status'] in ['done', 'error', 'timeout']:
                    break
            
            self.logger.info(f'Parsing Test Cases')
            if response['status'] == 'done':
                test_cases = self.test_case_construction(response['output_dict']['stdout'])
                return test_cases, test_case_generator_code, test_case_construction
            return []
        except Exception as e:
            self.logger.error(f'Generation Error: {e}')
            return []