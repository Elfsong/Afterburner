import json
from tqdm import tqdm
from datasets import load_dataset, Dataset


data = list()
for index in tqdm(range(100)):
    try:
        ds = load_dataset("Elfsong/APPS_New", "verified", split=f'{index}_{index+1}')
        data.extend(ds)
    except Exception as e:
        print(f"Error loading dataset for index {index}: {e}")

ds = Dataset.from_list(data)
ds.push_to_hub('Elfsong/APPS_python_verified', split='verified', private=True)


