# coding: utf-8

# Author: Du Mingzhe
# Date: 2025-04-11
# Description: Prepare data for GRPO training

import sys
from huggingface_hub import InferenceClient
sys.path.append("/home/mingzhe/Projects/Afterburner")

import utils
import os
from openai import OpenAI
from tqdm import tqdm
from datasets import load_dataset, Dataset, DatasetDict

GENERATION_TEMPLATE = """
## Instructions
You are an expert competitive programmer who excels at solving algorithm problems in multiple programming languages.
Your task is to implement a solution to the following problem in {target_lang}.

## Problem Description
{question}

## Starter Code
{starter_code}

## Output Format
- Provide the complete and efficient solution in **one markdown code block** with the correct language identifier.
- Strictly follow the function signature (name, parameters, etc.) as defined in the provided starter code.
- Include any necessary additional classes or helper functions outside of the provided starter code to ensure a fully functional solution.
"""

def cot_generation(question:str, starter_code:str):
    model_name = "Qwen/QwQ-32B-fast"
    prompt = GENERATION_TEMPLATE.format(target_lang="python3", question=question, starter_code=starter_code)

    client = OpenAI(
        base_url="https://api.studio.nebius.com/v1/",
        api_key="eyJhbGciOiJIUzI1NiIsImtpZCI6IlV6SXJWd1h0dnprLVRvdzlLZWstc0M1akptWXBvX1VaVkxUZlpnMDRlOFUiLCJ0eXAiOiJKV1QifQ.eyJzdWIiOiJnb29nbGUtb2F1dGgyfDEwMjM1NjEzMDM1MDE1MTM1ODEwMiIsInNjb3BlIjoib3BlbmlkIG9mZmxpbmVfYWNjZXNzIiwiaXNzIjoiYXBpX2tleV9pc3N1ZXIiLCJhdWQiOlsiaHR0cHM6Ly9uZWJpdXMtaW5mZXJlbmNlLmV1LmF1dGgwLmNvbS9hcGkvdjIvIl0sImV4cCI6MTkwMTg1NDA0MywidXVpZCI6IjMxMWI0ZmNhLWY5YjEtNGRhNi1iMjdkLWVhNWI0MjkyYmRkYyIsIm5hbWUiOiJNaW5nemhlIiwiZXhwaXJlc19hdCI6IjIwMzAtMDQtMDhUMDQ6NDc6MjMrMDAwMCJ9.dGAThMLRU7L-0dimkm0YiSRWRHGTR04Kk_EhPQSyP9k"
    )

    response = client.chat.completions.create(
        model=model_name,
        temperature=0.8,
        messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    )

    return response.choices[0].message.content


def cot_construction_pipeline(data_split="100%"):
    ds = load_dataset("Elfsong/Venus_Python", split=f"train[:{data_split}]")

    reasoning_responses = list()
    for instance in tqdm(ds):
        try:
            question = instance["question_content"]
            starter_code = instance["code_prompt"]
            llm_response = cot_generation(question, starter_code)
            reasoning_responses.append({
                "question_id": instance["problem_id"],
                "question_content": question,
                "code_prompt": starter_code,
                "llm_response": llm_response
            })
        except Exception as e:
            print(f"Error: {e}")
            continue

    reasoning_ds = Dataset.from_list(reasoning_responses)
    reasoning_ds.push_to_hub("Elfsong/Venus_Python_GRPO_Reasoning_Cold_Start")        

if __name__ == "__main__":
    cot_construction_pipeline(data_split="100%")