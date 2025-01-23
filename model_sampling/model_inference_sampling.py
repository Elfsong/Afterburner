# code: utf-8

# Author: Du Mingzhe (mingzhe@nus.edu.sg)
# Date: 2025-01-21

import sys
sys.path.append('/home/mingzhe/Projects/Afterburner')

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import utils
import client
from tqdm import tqdm
from datasets import load_dataset, Dataset

task_num = 4
temperature = 0.7
batch_number, batch_size = 8, 4

task_ds = load_dataset("codeparrot/apps", "competition", split='test')
model = client.HFClient("microsoft/Phi-3.5-mini-instruct")

for instance in tqdm(list(task_ds)[:task_num]):    
    try:
        # Extract instance details
        problem_content = instance['question']
        prompt = utils.pure_solution_generation_prompt_construct(problem_content, "")
        
        # Temperature Sampling
        solution_responses = list()
        
        for b in tqdm(range(batch_number)):    
            try:        
                # responses = model.json_generate(utils.CodeGeneration, prompt, k=batch_size, temperature=temperature)
                responses = model.text_generate(prompt, k=batch_size, temperature=temperature)
                
                for response in responses:
                    # code = utils.extract_json_from_response(response)
                    # if code:
                    solution_responses.append({
                        'problem_id': instance['problem_id'],
                        'difficulty': instance['difficulty'],
                        "code": response,
                    })
           
            except Exception as e:
                print(f"Error: {e}")
            
        eval_ds = Dataset.from_list(responses)
        eval_ds.push_to_hub("Elfsong/apps_generation", instance['problem_id'], split=f'temperature_{temperature}')
            
    except Exception as e:
        print(e)
        







