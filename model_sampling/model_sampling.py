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
import argparse
from tqdm import tqdm
from datasets import load_dataset, Dataset

parser = argparse.ArgumentParser(description='Model Evaluation Script')
parser.add_argument('--task_num', type=int, default=128, help='Number of tasks')
parser.add_argument('--temperature', type=float, default=0.7, help='Temperature for model generation')
parser.add_argument('--batch_number', type=int, default=8, help='Batch number')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
parser.add_argument('--model_path', type=str, default="microsoft/Phi-3.5-mini-instruct", help='Path to the model')
parser.add_argument('--model_name', type=str, default="Phi_3_5_mini_instruct_vanilla", help='Name of the model')
parser.add_argument('--difficulty', type=str, default="competition", help='Difficulty of the task')
parser.add_argument('--split', type=str, default='train', help='Dataset split')
args = parser.parse_args()

task_ds = load_dataset("codeparrot/apps", args.difficulty, split=args.split)
model = client.HFClient(args.model_path)

for index, instance in enumerate(list(task_ds)[:args.task_num]):   
    print(f"🚧 Task {index+1}/{args.task_num}")
    start_time = time.time()
    try:
        # Extract instance details
        problem_content = instance['question']
        prompt = utils.pure_solution_generation_prompt_construct(problem_content, "", "python")
        
        # Temperature Sampling
        solution_responses = list()
        
        for b in tqdm(range(args.batch_number), desc=f"Batch Inference"):  
            try:        
                responses = model.text_generate(prompt, k=args.batch_size, temperature=args.temperature)
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
        
        print(f"📦 Pushing to Hub...")
        eval_ds = Dataset.from_list(solution_responses)
        eval_ds.push_to_hub("Elfsong/apps_generation", f'{args.model_name}_temperature_{args.temperature}', split=str(instance['problem_id']))
    except Exception as e:
        print(e)
        
    end_time = time.time()
    print(f"⌛️ Time: {end_time - start_time:.2f} s")
    print("=============================================")
    
        







