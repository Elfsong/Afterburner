import json
from tqdm import tqdm
from pathlib import Path
from datasets import Dataset

problem_dict = dict()

# Problems Meta Info
for problem_file in tqdm(Path("./questions_clean").iterdir(), desc="[+] Processing Problem [Meta] Info..."):
    if 'extra.json' in problem_file.name: continue

    with open(problem_file, "r") as f:
        data = json.loads(f.read())

        problem_meta_info = {
            'problem_id': int(data['meta']['question_id']),
            'title': data['meta']['title_slug'],
            'question_content': data['question_md'],
            'difficulty': data['difficulty'],
            'tags': data['tags'],
            'code_prompt': data['starter_code'],
            'solutions': dict()
        }

        # Add solutions for each language
        for lang in data['solutions']:
            if data['solutions'][lang]:
                problem_meta_info['solutions'][lang] = [data['solutions'][lang]['code']]

        problem_dict[problem_meta_info['problem_id']] = problem_meta_info

# Problems Extra Info
for problem_file in tqdm(Path("./questions_clean").iterdir(), desc="[+] Processing Problem [Extra] Info..."):
    if 'extra.json' not in problem_file.name: continue

    with open(problem_file, "r") as f:
        problem_id = int(problem_file.name.split('_')[0])
        if problem_id not in problem_dict: continue

        data = json.loads(f.read())

        # Add successful submissions to the problem meta info
        for lang in data['submissions']:
            submission = data['submissions'][lang]
            if submission is None: continue

            code = submission['code']
            status_code = submission['statusCode']

            if lang not in problem_dict[problem_id]['solutions']:
                problem_dict[problem_id]['solutions'][lang] = []
            problem_dict[problem_id]['solutions'][lang].append(code)
        
        # Add additional solution to the problem meta info
        for solution in data['article_solutions']:
            if solution['lang'] not in problem_dict[problem_id]['solutions']:
                problem_dict[problem_id]['solutions'][solution['lang']] = []
            problem_dict[problem_id]['solutions'][solution['lang']].append(solution['code'])

# Problems Test Case Info
for problem_file in tqdm(Path("./data/leetcode").iterdir(), desc="[+] Processing Problem [Test Case] Info..."):
    problem_id = int(problem_file.name.split('_')[0])
    with open(problem_file, "r") as f:
        data = json.loads(f.read())
        test_case_generator = data['test_case_generator']
        test_case_evaluator = data['evaluator']
        test_case_runners = data['test_runners']
        test_cases = data['generated_tests']

        problem_dict[problem_id]['test_case_generator'] = test_case_generator
        problem_dict[problem_id]['test_case_evaluator'] = test_case_evaluator
        problem_dict[problem_id]['test_case_runners'] = test_case_runners
        problem_dict[problem_id]['test_cases'] = test_cases

# Save the processed data to HF
problem_list = list(problem_dict.values())
problem_ds = Dataset.from_list(problem_list)
problem_ds.push_to_hub("Elfsong/leetcode_data", num_shards=100, private=True)

print("[+] Done")