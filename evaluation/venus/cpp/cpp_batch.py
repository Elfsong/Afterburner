
import sys
sys.path.append("/home/mingzhe/Projects/Afterburner")

import os
os.environ["OPENAI_TOKEN"] = "API_TOKEN"

import json
import random
import pathlib
from tqdm import tqdm
from datasets import load_dataset
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_TOKEN"])

TEMPLATE = """
**Task**

Create a C++ test template with two specific placeholders:

1. ==Solution Code== : This placeholder will be replaced with the candidate's solution code (as a string).
2. ==Test Cases== : This placeholder will be replaced with a JSON-serialized list of test cases (as a string).

The final template should:

- Compile (g++ 11.2) and execute each test case using the provided solution.
- Output "PASS" if all test cases are successful; otherwise, output "FAILED".

**Important**

- Use the placeholders **exactly** as written above; they will be programmatically substituted via:

```python
test_template = test_template.replace("==Solution Code==", example_solution_code)
test_case_str = json.dumps(test_cases, separators=(',', ':'))
test_case_str_literal = f'R"({{test_case_str}})"'
test_template = test_template.replace("==Test Cases==", test_case_str_literal)
```

- Provide **NO** explanations or code comments.
- Don't include third-party libraries.
- Respond MERELY with the template code in **one markdown code block** with appropriate language identifier ````cpp{{template_code}}```.
- Place '==Solution Code==' and '==Test Cases==' in the global namespace; Do **NOT** wrap them in any class or function:

```cpp
==Solution Code==

Solution sol;
string test_cases_str = ==Test Cases==;
```

## Problem Content
{problem_content}

## Example Solution Code 1
{solution_code_1}

## Example Solution Code 2
{solution_code_2}

## Example Test Cases
test_cases = [{{'input': 'input_str', 'output': 'output_str'}}, {{'input': 'input_str', 'output': 'output_str'}}]
"""

def create_batch_case(instance):
    problem_id = instance['question_id']
    solutions = instance['rt_list'] + instance['mm_list']
    if problem_id not in leetcode_dict: return None
    if not leetcode_dict[problem_id]['test_cases']: return None
    if not solutions: return None

    problem_content = leetcode_dict[problem_id]['question_content']
    # test_cases = json.loads(leetcode_dict[problem_id]['test_cases'])[:2]
    example_solution_code_1 = random.choice(solutions)['code']
    example_solution_code_2 = random.choice(solutions)['code']

    prompt = TEMPLATE.format(problem_content=problem_content, solution_code_1=example_solution_code_1, solution_code_2=example_solution_code_2)

    batch_case = {
        "custom_id": f"request-{problem_id}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "o4-mini-2025-04-16",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_completion_tokens": 8192
        }
    }
    return batch_case

correct_problem_ids = list()
for file in pathlib.Path("./correct_templates").glob("*.cpp"):
    correct_problem_ids.append(int(file.name.split("_")[2].split(".")[0]))
print(f"Correct Problems: {len(correct_problem_ids)}")

leetcode_dict = dict()
leetcode_ds = load_dataset("Elfsong/leetcode_data")
for instance in leetcode_ds['train']:
    leetcode_dict[instance['problem_id']] = instance
venus_ds = load_dataset("Elfsong/Venus", "cpp", split=f"train")

# create batch cases
batch_cases = list()
for instance in tqdm(venus_ds):
    if instance['question_id'] in correct_problem_ids: continue
    batch_case = create_batch_case(instance)
    if batch_case is None: continue
    batch_cases.append(batch_case)

# write batch cases to file
with open(f"batchinput.jsonl", "w") as f:
    for batch_case in batch_cases:
        f.write(json.dumps(batch_case) + "\n")





