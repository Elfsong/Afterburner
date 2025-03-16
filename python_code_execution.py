# code: utf-8

# Author: Du Mingzhe (mingzhe@nus.edu.sg)
# Date: 2025-01-13

import sys
sys.path.append('/home/nus_cisco_wp1/Projects/llm-sandbox')

import json
import utils
from tqdm import tqdm
from random import sample
from llm_sandbox import SandboxSession
from datasets import load_dataset, Dataset

task_ds = load_dataset("Elfsong/Venus_t", "python3", split="train")

for instance in tqdm(list(task_ds)[300:]):
    eval_ds = load_dataset("Elfsong/Venus_t", "python3", split="eval")
    
    try:
        content = instance['content']
        libraries = instance['libraries']
        question_id = instance['question_id']
        entry_point = instance['entry_point']
        import_code = instance['import_code']
        solution_prompt = instance['code_prompt']
        test_cases = instance['test_case']
        
        if not test_cases:
            continue
        
        if question_id in [eval_instance['question_id'] for eval_instance in eval_ds]:
            continue

        test_cases = instance['test_case'].split("\n")
        test_cases = [json.loads(test_case) for test_case in test_cases if test_case]
        test_case_generator = instance['test_case_generator']
        instance = utils.solution_filter(instance)
        solutions = instance['rt_list'] + instance['mm_list']
        
        solution_t, solution_c = 0, 0
        for solution in tqdm(sample(solutions, 3)):
            try:
                solution_t += 1
                execution_code = utils.code_evalution_construct(import_code, solution['code'], test_case_generator, entry_point, test_cases)
                with SandboxSession(lang="python", verbose=False) as session:
                    session.setup(libraries=instance['libraries'])
                    response = utils.run_with_timeout(session.run, 60, code=execution_code, run_memory_profile=True)
                    status = response['output']['stdout'].strip() if response['output']['stdout'] else "failed"
                    if status == "success":
                        solution_c += 1
                        print("Bingo!")
                    else:
                        print(response['output']['stderr'])
                        break
            except Exception as e:
                print(e)
        
        print("Question ID: {}, Total Solutions: {}, Correct Solutions: {}".format(question_id, solution_t, solution_c))
        if solution_t > 0 and solution_t == solution_c:
            eval_list = list(eval_ds)
            eval_list.append(instance)
            eval_ds = Dataset.from_list(eval_list)
            eval_ds.push_to_hub("Elfsong/Venus_t", 'python3', split='eval')
            eval_ds.cleanup_cache_files()
            print("\n### Get a new instance!###\n")
            
    except Exception as e:
        print(e)
        







