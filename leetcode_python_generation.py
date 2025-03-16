# coding: utf-8

# Just-In-Time Tester (JITT)
# Author: Du Mingzhe (mingzhe@nus.edu.sg)
# Date: 2025-03-16


from tqdm import tqdm
from jitt import JITTCaller
from datasets import load_dataset, Dataset

if __name__ == "__main__":
    jitt_caller = JITTCaller(number_of_workers=25)
    
    ds = load_dataset("Elfsong/JITT", "python3")
    
    dataset_dict = dict()
    code_solutions = list()
    for instance in ds['train']:
        dataset_dict[instance['problem_id']] = instance
        if instance['test_cases'] == "":
            code_solutions.append({
                'solution_code': instance['solution_code'],
                'problem_id': instance['problem_id']
            }
        )
    print(f"Number of code solutions that need to update: {len(code_solutions)}")
    print(f"Number of instances: {len(dataset_dict)}")
    
    for i in tqdm(range(0, len(code_solutions), 100)):
        try:
            results = jitt_caller.batch_generate(code_solutions[i:i+100])
            
            for result in results:
                promblem_id = result['problem_id']
                
                if result['test_cases']:
                    print(f"[{promblem_id}] success!")
                    dataset_dict[promblem_id]['test_cases'] = result['test_cases']
                    dataset_dict[promblem_id]['test_case_generator_code'] = result['test_case_generator_code'] if result['test_case_generator_code'] else dataset_dict[promblem_id]['test_case_generator_code']
                    dataset_dict[promblem_id]['libraries'] = result['libraries'] if result['libraries'] else dataset_dict[promblem_id]['libraries']
                    dataset_dict[promblem_id]['import_statements'] = result['import_statements'] if result['import_statements'] else dataset_dict[promblem_id]['import_statements']
                    dataset_dict[promblem_id]['executable_code'] = result['executable_code'] if result['executable_code'] else dataset_dict[promblem_id]['executable_code']
                else:
                    print(f"[{promblem_id}] failed!")
                    
            dataset_set = Dataset.from_list(list(dataset_dict.values()))
            dataset_set.push_to_hub("Elfsong/JITT", 'python3')

        except Exception as e:
            print(e)
    
