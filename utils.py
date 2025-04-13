# code: utf-8

# Author: Du Mingzhe (mingzhe@nus.edu.sg)
# Date: 2025-01-13

import os
import re
import json
import bisect
import threading
from typing import List
from openai import OpenAI
from functools import cache
from pydantic import BaseModel, ConfigDict
from huggingface_hub import InferenceClient

TOKEN_REGISTRY = {
    "nebius": os.getenv("NEBIUS_TOKEN"),
    "together": os.getenv("TOGETHER_TOKEN"),
    "huggingface": os.getenv("HUGGINGFACE_TOKEN"),
    "openai": os.getenv("OPENAI_TOKEN"),
    "claude": os.getenv("CLAUDE_TOKEN"),
    "local": "NONE",
}

LANGUAGE_REGISTRY = {
    'cpp': {"id": 0, "verbose_name": "C++", "md_langs": ["cpp", "c++"]},
    'java': {"id": 1, "verbose_name": "Java", "md_langs": ["java"]},
    'python': {"id": 2, "verbose_name": "Python", "md_langs": ["python", "py"]},
    'mysql': {"id": 3, "verbose_name": "MySQL", "md_langs": ["mysql", "sql"]},
    'c': {"id": 4, "verbose_name": "C", "md_langs": ["c"]},
    'csharp': {"id": 5, "verbose_name": "C#", "md_langs": ["cs", "csharp", "c#"]},
    'javascript': {"id": 6, "verbose_name": "JavaScript", "md_langs": ["js", "javascript", "node"]},
    'ruby': {"id": 7, "verbose_name": "Ruby", "md_langs": ["rb", "ruby", "jruby", "macruby", "rake", "rbx"]},
    'bash': {"id": 8, "verbose_name": "Bash", "md_langs": ["sh", "bash", "shell", "shell-script", "zsh"]},
    'swift': {"id": 9, "verbose_name": "Swift", "md_langs": ["swift"]},
    'golang': {"id": 10, "verbose_name": "Go", "md_langs": ["golang", "go"]},
    'python3': {"id": 11, "verbose_name": "Python3", "md_langs": ["python", "py"]},
    'scala': {"id": 12, "verbose_name": "Scala", "md_langs": ["scala"]},
    'kotlin': {"id": 13, "verbose_name": "Kotlin", "md_langs": ["kotlin"]},
    'mssql': {"id": 14, "verbose_name": "MS SQL Server", "md_langs": ["tsql", "mssql"]},
    'oraclesql': {"id": 15, "verbose_name": "Oracle", "md_langs": ["plsql", "oraclesql"]},
    'rust': {"id": 18, "verbose_name": "Rust", "md_langs": ["rust", "rs"]},
    'php': {"id": 19, "verbose_name": "PHP", "md_langs": ["php"]},
    'typescript': {"id": 20, "verbose_name": "TypeScript", "md_langs": ["ts", "typescript"]},
    'racket': {"id": 21, "verbose_name": "Racket", "md_langs": ["racket"]},
    'erlang': {"id": 22, "verbose_name": "Erlang", "md_langs": ["erlang"]},
    'elixir': {"id": 23, "verbose_name": "Elixir", "md_langs": ["elixir"]},
    'dart': {"id": 24, "verbose_name": "Dart", "md_langs": ["dart"]},
    'pythondata': {"id": 25, "verbose_name": "Pandas", "md_langs": ["pandas", "pythondata"]},
    'react': {"id": 26, "verbose_name": "React", "md_langs": ["jsx", "react"]},
    'vanillajs': {"id": 27, "verbose_name": "Vanilla JS", "md_langs": ["js", "javascript", "vanillajs"]},
    'postgresql': {"id": 28, "verbose_name": "PostgreSQL", "md_langs": ["postgres", "postgresql", "pgsql"]},
    'cangjie': {"id": 29, "verbose_name": "Cangjie", "md_langs": ["cangjie"]},
}

@cache
def get_token(provider_name: str) -> str | None:
    """
    Returns the token for the specified provider name from TOKEN_REGISTRY.
    Returns None if not found.
    """
    return TOKEN_REGISTRY.get(provider_name)

@cache
def get_provider_name(provider_name: str) -> str | None:
    """
    Returns the provider name for the specified token from TOKEN_REGISTRY.
    Returns None if not found.
    """
    provider_list = ['black-forest-labs', 'cerebras', 'cohere', 'fal-ai', 'fireworks-ai', 'hf-inference', 'hyperbolic', 'nebius', 'novita', 'openai', 'replicate', 'sambanova', 'together']
    return provider_name if provider_name in provider_list else None

@cache
def get_url(provider_name: str) -> str | None:
    """
    Returns the URL for the specified provider name from TOKEN_REGISTRY.
    Returns None if not found.
    """
    if provider_name == "claude":
        return "https://api.anthropic.com/v1/"
    elif provider_name == "local":
        return "http://localhost:8000/v1"
    elif provider_name == "nebius":
        return "https://api.studio.nebius.com/v1/"
    else:
        return None

@cache
def get_md_lang(lang: str) -> str | None:
    """
    Returns the first Markdown code block identifier for the given language key from LANG_LOOKUP.
    Returns None if not found.
    """
    md_langs = LANGUAGE_REGISTRY.get(lang, {}).get("md_langs", [])
    return md_langs[0] if md_langs else None

@cache
def get_lang_by_md_lang(md_lang: str) -> str | None:
    """
    Returns the language key for the given Markdown code block identifier from LANG_LOOKUP.
    Returns None if not found.
    """
    if md_lang in LANGUAGE_REGISTRY["python3"]["md_langs"]:
        return "python3"
    return next((key for key, value in LANGUAGE_REGISTRY.items() if md_lang in value["md_langs"]), None)

@cache
def get_lang_by_verbose_name(verbose_name: str) -> str | None:
    """
    Returns the language key for the given verbose name from LANG_LOOKUP.
    Returns None if not found.
    """
    return next((key for key, value in LANGUAGE_REGISTRY.items() if verbose_name.lower() == value["verbose_name"].lower()), None)

def extract_code_blocks(text: str) -> list[dict[str, str]]:
    _CODE_BLOCK_PATTERN = re.compile(r"```([\w+-]*)(?:\n|\r\n)?(.*?)```", re.DOTALL)
    blocks: list[dict[str, str]] = []
    for match in _CODE_BLOCK_PATTERN.finditer(text):
        lang = match.group(1).strip()
        code = match.group(2).strip()
        blocks.append({"lang": lang, "code": code})
    return blocks

def wrap_code_block(lang: str, code: str) -> str:
    return f"```{get_md_lang(lang)}\n{code}\n```"

def percentage_position(num: float, lst: list[float]) -> float:
    if not lst:
        return 0.0
    sorted_lst = sorted(lst)
    insert_pos = bisect.bisect_left(sorted_lst, num)
    percentage = 1 - (insert_pos / len(lst))
    return percentage

def model_inference(inference_provider, model_name, prompt, temperature, max_tokens):
    if inference_provider == "openai" and model_name == "o3-mini":
        client = OpenAI(api_key=get_token(inference_provider))
        response = client.responses.create(
            model=model_name,
            input=[{"role": "user","content": [{"type": "input_text", "text": prompt}]}],
            text={"format": {"type": "text"}},
            reasoning={"effort": "low"},
            tools=[],
            max_completion_tokens=max_tokens
        )
        return response.output_text
    elif inference_provider == "local":
        client = OpenAI(base_url=get_url(inference_provider), api_key=get_token(inference_provider))
        completion = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user","content": prompt}],
            temperature=temperature,
            max_completion_tokens=max_tokens
        )
        return completion.choices[0].message.content
    else:
        # Prepare the API client
        client = InferenceClient(
            provider=get_provider_name(inference_provider),
            api_key=get_token(inference_provider), 
            base_url=get_url(inference_provider),
        )
        
        # Generate the solution
        completion = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )

        return completion.choices[0].message.content

def extract_solution_call(cpp_code: str) -> str:
    pattern = re.compile(r'\bsol\.(\w+)\s*\(([^)]*)\)', re.DOTALL)
    match = pattern.search(cpp_code)
    if match:
        func_name = match.group(1)
        args = match.group(2).replace("\n", " ").strip()
        return f"{func_name}({args})"
    return ""

def extract_cpp_functions(source_code: str, function_names: list) -> dict:
    functions = {}
    pattern = re.compile(rf'\b(?:[\w:<>,&*]+\s+)+({"|".join(function_names)})\s*\(([^)]*)\)\s*{{')

    pos = 0
    while pos < len(source_code):
        match = pattern.search(source_code, pos)
        if not match:
            break
        func_name = match.group(1)
        start = match.start()
        brace_count = 0
        i = match.end() - 1  # Start from the opening brace

        # Scan forward to find matching closing brace
        while i < len(source_code):
            if source_code[i] == '{':
                brace_count += 1
            elif source_code[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    func_body = source_code[start:i+1]
                    functions[func_name] = func_body.strip()
                    pos = i + 1
                    break
            i += 1
        else:
            # Unbalanced braces
            break

    return functions
