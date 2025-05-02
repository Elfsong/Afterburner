# coding: utf-8

# Author: Du Mingzhe (mingzhe@nus.edu.sg)
# Date: 2025-04-26

import sys
sys.path.append("/home/mingzhe/Projects/Afterburner")

from datasets import load_dataset, Dataset
from tqdm import tqdm
import random
import numpy as np
import scipy.stats as stats
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="gpt_4o", help="Name of the model to evaluate")
args = parser.parse_args()

model_name = args.model_name
bootstrap_size = 4
bootstrap_iterations = 128

ds = load_dataset("Elfsong/Venus_Python_Model_Evaluation", model_name, split="train")
print(f"Current Dataset: {model_name}")

# Data Preprocessing
problem_dict = dict()
for problem in tqdm(ds):
    if problem["problem_id"] not in problem_dict:
        problem_dict[problem["problem_id"]] = [problem]
    else:
        problem_dict[problem["problem_id"]].append(problem)

# Bootstrap Sampling
bootstrap_time_list = list()
bootstrap_memory_list = list()
bootstrap_integral_list = list()
bootstrap_pass_list = list()

for _ in range(bootstrap_iterations):
    # Sampling {bootstrap_size} solutions from each problem
    bootstrap_ds = list()
    for problem_id, problems in problem_dict.items():
        bootstrap_ds.extend(random.sample(problems, min(len(problems), bootstrap_size)))
    
    # Evaluate the performance of the bootstrap dataset
    scores = {"total_c": 0, "pass_c": 0, "time_s": 0,"memory_s": 0, "integral_s": 0}

    for instance in bootstrap_ds:
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

    bootstrap_time_list.append(scores["time_score"])
    bootstrap_memory_list.append(scores["memory_score"])
    bootstrap_integral_list.append(scores["integral_score"])
    bootstrap_pass_list.append(scores["pass_score"])

def calculate_confidence_interval(bootstrap_list, name=""):
    print(f"Current {name} Bootstrap Dataset: {len(bootstrap_list)}")
    samples = np.array(bootstrap_list)
    n = len(samples)
    mean = np.mean(samples)
    std = np.std(samples, ddof=1)
    sem = std / np.sqrt(n)


    confidence = 0.95
    t_score = stats.t.ppf((1 + confidence) / 2, df=n-1)
    margin_of_error = t_score * sem

    ci_lower = mean - margin_of_error
    ci_upper = mean + margin_of_error

    print(f"Mean: {mean:.4f}")
    print(f"Std: {std:.4f}")
    print(f"Sem: {sem:.4f}")
    print(f"CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"Result: {mean*100:.2f}% [{ci_lower*100:.2f}% {ci_upper*100:.2f}%]")


calculate_confidence_interval(bootstrap_time_list, "Time")
calculate_confidence_interval(bootstrap_memory_list, "Memory")
calculate_confidence_interval(bootstrap_integral_list, "Integral")
calculate_confidence_interval(bootstrap_pass_list, "Pass")












