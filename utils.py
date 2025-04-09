# code: utf-8

# Author: Du Mingzhe (mingzhe@nus.edu.sg)
# Date: 2025-01-13

import os
import re
import json
import bisect
import threading
from typing import List
from functools import cache
from pydantic import BaseModel, ConfigDict

TOKEN_REGISTRY = {
    "neibus": os.getenv("NEIBUS_TOKEN"),
    "together": os.getenv("TOGETHER_TOKEN"),
    "huggingface": os.getenv("HUGGINGFACE_TOKEN"),
    "openai": os.getenv("OPENAI_TOKEN"),
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