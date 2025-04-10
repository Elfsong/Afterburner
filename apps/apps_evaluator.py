# coding: utf-8

# Author: Du Mingzhe (mingzhe@nus.edu.sg)
# Date: 2025-03-19

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
from monolith import monolith
from datasets import load_dataset, Dataset
from multiprocessing.dummy import Pool as ThreadPool

GENERATION_TEMPLATE = """
## Instructions
You are an expert competitive programmer who excels at solving algorithm problems in multiple programming languages.
Your task is to implement a solution to the following problem in {target_lang}.

## Problem Description
{question}

## Starter Code
{starter_code}

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
        self.assertEqual(expected, actual)
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

class AppsEvaluator:
    def __init__(self, lang, monolith_timeout, case_multiply, number_of_workers):
        self.lang = lang
        self.monolith_timeout = monolith_timeout
        self.case_multiply = case_multiply
        self.number_of_workers = number_of_workers
        self.monolith_client = monolith.Monolith(backend_url='https://monolith.cool', retries=3)
    
    @classmethod
    def apps_async_evaluation(cls, solution_code: str, test_cases: List, timeout: int) -> dict:
        response = {
            'passed': False, 
            'time': float('inf'), 
            'memory': float('inf'), 
            'integral': float('inf'), 
            'status': 'error',
        }
        
        try:            
            monolith_client = monolith.Monolith(backend_url='https://monolith.cool', retries=3)
            
            # Construct Test Code
            solution_code = textwrap.indent(solution_code.strip(), "\t")
            test_case_list_str = json.dumps(test_cases, indent=4)
            test_code = EVALUATION_TEMPLATE.format(solution_code=solution_code, test_case_list=test_case_list_str, case_multiply=100)
            
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

            # Update Response
            if result["status"] == "done":
                response['passed'] = True if result['output_dict']['stdout'] == 'Success\n' else False
                response['time'] = result['output_dict']['duration']
                response['memory'] = result['output_dict']['peak_memory']
                response['integral'] = result['output_dict']['integral']
        except Exception as e:
            response['status'] = 'error'
            print(f"Evaluation Error: {e}", e.with_traceback(None))
        finally:
            return response
    
    @classmethod
    def apps_sync_evaluation(cls, solution_code: str, instance: Any, case_multiply: int, timeout: int) -> dict:
        response = {'passed': False, 'time': float('inf'), 'memory': float('inf'), 'integral': float('inf'), 'status': 'error'}
        try:
            # Construct Test Code
            solution_code = textwrap.indent(solution_code.strip(), "\t")
            test_case_list_str = instance['test_cases']
            test_code = EVALUATION_TEMPLATE.format(solution_code=solution_code, test_case_list=test_case_list_str, case_multiply=100)
            
            # with open(f"test_code_{instance['problem_id']}.py", "w") as f:
            #     f.write(test_code)
            
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
    def apps_generation(cls, inference_provider, model_name, instance: Any, target_lang: str, temperature=0, max_token=4096) -> dict:
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
    
    def apps_evaluation_pipeline(self, model_name, dataset_split_name, inference_provider, data_precentage="100%", data_multiply=1, mode="G+E"):            
        print(f"[+] Processing Model: {model_name} [{dataset_split_name}] in [{data_precentage}] - {data_multiply} - {mode} - {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
        # Generating solutions (G)
        if mode in ["G", "G+E"]:
            apps_dataset = load_dataset("Elfsong/APPS_Verfied", split=f"train[:{data_precentage}]")
            test_packs = list()
            for instance in tqdm(apps_dataset, desc='Generating solutions'):
                try:
                    generated_solution = self.apps_generation(inference_provider, model_name, instance, self.lang, temperature=0, max_token=16384)
                except Exception as e:
                    print(f"[-] Generation Error: {e}")
                    generated_solution = ""
                finally:
                    test_packs.append({"problem_id": int(instance['problem_id']), "solution": generated_solution})
            ds = Dataset.from_list(test_packs)
            ds.push_to_hub("Elfsong/APPS_Model_Evaluation", dataset_split_name, private=True)
        
        # Parallel Evaluation (E)
        if mode in ["E", "G+E"]:
            venus_dataset = load_dataset("Elfsong/APPS_Verfied", split=f"train[:{data_precentage}]")
            code_generation_dataset = load_dataset("Elfsong/APPS_Model_Evaluation", dataset_split_name, split="train")
            test_packs = [(code['solution'], instance, self.case_multiply, self.monolith_timeout) for code, instance in zip(code_generation_dataset, venus_dataset)]
            test_packs = test_packs * data_multiply
                
            results = list()
            with ThreadPool(self.number_of_workers) as pool:
                with tqdm(total=len(test_packs), desc='Solution Evaluation') as pbar:
                    for result in pool.imap(lambda args: AppsEvaluator.apps_sync_evaluation(*args), test_packs):
                        results.append(result)
                        pbar.update(1)

            # Score Calculation
            instance_list = list()
            for instance, test_pack, result in zip(venus_dataset.repeat(data_multiply), test_packs, results):
                solutions = json.loads(instance['solutions'])
                time_distribution = [s['time'] for s in solutions if s['passed']]
                memory_distribution = [s['memory'] for s in solutions if s['passed']]
                integral_distribution = [s['integral'] for s in solutions if s['passed']]

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

            print(f"[{dataset_split_name}] Pass@1:{scores['pass_score']:.2f} Time_Precent:{scores['time_score']:.2f} Memory_Precent:{scores['memory_score']:.2f} Integral_Precent:{scores['integral_score']:.2f}")
                
            # Save the results
            ds = Dataset.from_list(instance_list)
            ds.push_to_hub("Elfsong/APPS_Python_Model_Evaluation", dataset_split_name, private=True)
        print("========================================================")
    
    def apps_distribution_pipeline(self):
        for i in range(100):
            apps_data = load_dataset("Elfsong/APPS", 'default', split=f"test[{i}%:{(i+1)}%]")        
            new_apps_data = list()
            
            total_count = len(apps_data)
            for index, instance in enumerate(apps_data):
                problem_id = instance["problem_id"]
                problem_content = instance["question"]
                code_prompt = instance["starter_code"]
                difficulty = instance["difficulty"]
                solutions = json.loads(instance["solutions"])
                test_cases = json.loads(instance["test_cases"])
                
                # Prepare Test Packs (add code_prompt to each solution)
                test_packs = [(code_prompt + '\n' + solution, test_cases, self.monolith_timeout) for solution in solutions]
                
                print(f'[+] Problem {problem_id} [{index}/{total_count}] in [{i}% - {(i+1)}%] - {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')
                
                # Check Monolith Status
                while True:
                    queue_size = self.monolith_client.get_status().get('current_queue_size', -1)
                    if queue_size == 0:
                        break
                    time.sleep(5)
                
                # Parallel Evaluation
                results = list()
                with ThreadPool(self.number_of_workers) as pool:
                    with tqdm(total=len(test_packs), desc='Solution Evaluation') as pbar:
                        for result in pool.imap(apps_evaluation_unpacker, test_packs):
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
                for (solution, result) in zip(solutions, results):
                    verified_solutions.append({
                        'code': solution, 
                        'status': str(result['status']),
                        'passed': bool(result['passed']),
                        'time': float(result['time']),
                        'memory': float(result['memory']),
                        'integral': float(result['integral'])
                    })
                        
                # Save New APPS Data
                new_apps_data.append({
                    "problem_id": problem_id,
                    "problem_content": problem_content,
                    "code_prompt": code_prompt,
                    "difficulty": difficulty,
                    "solutions": json.dumps(verified_solutions),
                    "test_cases": json.dumps(test_cases),
                })
            
                print('============================================================')
            
            # Push New APPS Data to HF
            new_apps_dataset = Dataset.from_list(new_apps_data)
            new_apps_dataset.push_to_hub("Elfsong/APPS_New", 'verified', split=f"{i}_{(i+1)}")
            

if __name__ == "__main__":
    apps_evaluator = AppsEvaluator(lang="python3", monolith_timeout=90, case_multiply=1024, number_of_workers=81)
    # evaluator.apps_distribution_pipeline()
    
    apps_evaluator.apps_evaluation_pipeline(model_name="google/gemma-3-27b-it", dataset_split_name="gemma_3_27b", inference_provider="nebius", data_precentage="100%", data_multiply=16, mode="G")
    apps_evaluator.apps_evaluation_pipeline(model_name="meta-llama/Llama-3.3-70B-Instruct", dataset_split_name="llama_3_3_70b_instruct", inference_provider="together", data_precentage="100%", data_multiply=16, mode="G")
    apps_evaluator.apps_evaluation_pipeline(model_name="meta-llama/Llama-3.1-8B-Instruct", dataset_split_name="llama_3_1_8b_instruct", inference_provider="nebius", data_precentage="100%", data_multiply=16, mode="G")
    apps_evaluator.apps_evaluation_pipeline(model_name="meta-llama/Llama-3.1-405B-Instruct", dataset_split_name="llama_3_1_405b_instruct", inference_provider="nebius", data_precentage="100%", data_multiply=16, mode="G")
    apps_evaluator.apps_evaluation_pipeline(model_name="Qwen/Qwen2.5-Coder-32B-Instruct", dataset_split_name="qwen_2_5_coder_32b_instruct", inference_provider="nebius", data_precentage="100%", data_multiply=16, mode="G")
    apps_evaluator.apps_evaluation_pipeline(model_name="Qwen/Qwen2.5-Coder-7B-Instruct", dataset_split_name="qwen_2_5_coder_7b_instruct", inference_provider="nebius", data_precentage="100%", data_multiply=16, mode="G")
    apps_evaluator.apps_evaluation_pipeline(model_name="deepseek-ai/DeepSeek-V3-0324", dataset_split_name="deepseek_v3", inference_provider="nebius", data_precentage="100%", data_multiply=16, mode="G")
    apps_evaluator.apps_evaluation_pipeline(model_name="claude-3-7-sonnet-20250219", dataset_split_name="claude_3_7_sonnet", inference_provider="claude", data_precentage="100%", data_multiply=16, mode="G") 
    apps_evaluator.apps_evaluation_pipeline(model_name="deepseek-ai/DeepSeek-V3-0324", dataset_split_name="deepseek_v3", inference_provider="nebius", data_precentage="100%", data_multiply=16, mode="G")

