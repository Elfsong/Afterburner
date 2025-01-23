# code: utf-8

# Author: Du Mingzhe (mingzhe@nus.edu.sg)
# Date: 2025-01-21

import sys
sys.path.append('/home/mingzhe/Projects/Afterburner')

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import time
import utils
import client
from tqdm import tqdm
from datasets import load_dataset, Dataset

task_num = 128
temperature = 0.7
batch_number, batch_size = 8, 4
model_path = "microsoft/Phi-3.5-mini-instruct"
model_name = "Phi_3_5_mini_instruct_vanilla"

task_ds = load_dataset("codeparrot/apps", "competition", split='test')
model = client.HFClient(model_path)

for index, instance in enumerate(list(task_ds)[:task_num]):   
    print(f"Task {index+1}/{task_num}")
    start_time = time.time()
    try:
        # Extract instance details
        problem_content = instance['question']
        prompt = utils.pure_solution_generation_prompt_construct(problem_content, "", "python")
        
        # Temperature Sampling
        solution_responses = list()
        
        for b in tqdm(range(batch_number)):    
            try:        
                responses = model.text_generate(prompt, k=batch_size, temperature=temperature)
                
                for response in responses:
                    code = utils.extract_python_code_from_response(response)
                    if code:
                        solution_responses.append({
                            'problem_id': instance['problem_id'],
                            'difficulty': instance['difficulty'],
                            "code": code,
                        })
           
            except Exception as e:
                print(f"Error: {e}")
            
        eval_ds = Dataset.from_list(solution_responses)
        eval_ds.push_to_hub("Elfsong/apps_generation", f'{model_name}_temperature_{temperature}', split=str(instance['problem_id']))
    except Exception as e:
        print(e)
        
    end_time = time.time()
    print(f"Time: {end_time - start_time:.2f} s")
    
        







