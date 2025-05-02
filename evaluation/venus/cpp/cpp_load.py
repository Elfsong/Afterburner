import sys
sys.path.append("/home/mingzhe/Projects/Afterburner")

import json
import utils
import random
import pathlib
from datasets import load_dataset

test_templates = list()
for file_name in pathlib.Path("./batch_results/").glob("*.jsonl"):
    print(file_name)
    with open(file_name, "r") as f:
        for line in f.readlines():
            try:
                instance = json.loads(line)
                custom_id = instance['custom_id']
                problem_id = custom_id.split('-')[1]
                template = instance['response']['body']['choices'][0]['message']['content']
                template = template.replace("```cpp", "").replace("```", "")
                
                with open(f"cpp_templates/test_template_{problem_id}.cpp", "w") as f:
                    f.write(template)
            except Exception as e:
                print(e)
                continue
