import os
import json
import random
import pathlib
import requests
from tqdm import tqdm
from itertools import repeat
from datasets import load_dataset
from multiprocessing.dummy import Pool as ThreadPool


def venus_evaluation(problem_id: int, solution_code: str, timeout: int) -> dict:
    response = {'problem_id': problem_id, 'passed': False, 'time': float('inf'), 'memory': float('inf'), 'integral': float('inf'), 'status': 'error'}
    try:
        # Submit Test Code to Monolith
        data = {
            'code': solution_code,
            'language': 'cpp',
            'libraries': [],
            'timeout': timeout,
            'run_profiling': True
        }
        monolith_response = requests.post(f'https://monolith.cool/execute', json=data, timeout=(120, timeout))
        if monolith_response.status_code == 200:
            monolith_response = monolith_response.json()

            response['status'] = monolith_response['status']
            if monolith_response["status"] == "success":
                response['passed'] = True if monolith_response['output_dict']['stdout'] == 'PASS' else False
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

venus_ds = load_dataset("Elfsong/Venus", "cpp", split='train')
venus_dict = {}
for instance in venus_ds:
    venus_dict[int(instance['question_id'])] = instance

leetcode_ds = load_dataset("Elfsong/leetcode_data", split='train')
leetcode_dict = {}
for instance in leetcode_ds:
    leetcode_dict[int(instance['problem_id'])] = instance


for i in range(10):
    solution_codes = list()
    problem_ids = list()
    for template_name in pathlib.Path("./cpp_templates/").glob("*.cpp"):
        problem_id = int(template_name.stem.split("_")[2])

        if problem_id not in leetcode_dict: continue
        if problem_id not in venus_dict: continue

        leetcode_instance = leetcode_dict[problem_id]
        venus_instance = venus_dict[problem_id]

        solutions = venus_instance['rt_list'] + venus_instance['mm_list']
        if len(solutions) == 0: continue
        solution_code = random.choice(solutions)['code']

        test_cases = json.loads(leetcode_instance['test_cases'])
        test_case_str = json.dumps(test_cases, separators=(',', ':'))
        test_case_str_literal = f'R"({test_case_str})"'

        template_code = open(template_name, "r").read()
        template_code = template_code.replace("==Solution Code==", solution_code)
        template_code = template_code.replace("==Test Cases==", test_case_str_literal)

        solution_codes.append(template_code)
        problem_ids.append(problem_id)

    results = list()
    timeout = 60
    with ThreadPool(81) as pool:
        with tqdm(total=len(solution_codes), desc='Solution Evaluation') as pbar:
            for result in pool.imap(lambda args: venus_evaluation(*args), zip(problem_ids, solution_codes, repeat(timeout))):
                results.append(result)
                pbar.update(1)

    pass_c, total_c = 0, 0
    for result in results:
        if result['passed']:
            problem_id = result['problem_id']
            os.rename(f"cpp_templates/test_template_{problem_id}.cpp", f"correct_templates/test_template_{problem_id}.cpp")
            pass_c += 1
        total_c += 1
    print(f"Passed: {pass_c}/{total_c}")
