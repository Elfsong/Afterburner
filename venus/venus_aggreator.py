import json
from tqdm import tqdm
from datasets import load_dataset, Dataset


data = list()
for index in tqdm(range(100)):
    try:
        ds = load_dataset("Elfsong/Venus_python", "verified", split=f'{index}_{index+1}')
        for instance in tqdm(ds):
            data.append(instance)
    except Exception as e:
        print(f"Error loading dataset for index {index}: {e}")


ds = Dataset.from_list(data)
ds.push_to_hub('Elfsong/Venus_python_verified', split='verified', private=True)


