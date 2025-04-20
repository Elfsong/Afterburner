# code: utf-8

# Author: Du Mingzhe (mingzhe@nus.edu.sg)
# Date: 2025-01-13

import os
import re
import bisect
import anthropic
from openai import OpenAI
from huggingface_hub import InferenceClient

TOKEN_REGISTRY = {
    "nebius": os.getenv("NEBIUS_TOKEN"),
    "together": os.getenv("TOGETHER_TOKEN"),
    "huggingface": os.getenv("HUGGINGFACE_TOKEN"),
    "openai": os.getenv("OPENAI_TOKEN"),
    "claude": os.getenv("CLAUDE_TOKEN"),
    "local": "NONE",
}

_LANGUAGE_IMPORTS = {
    "python3": r"""
import re
from re import match, search, sub, split, findall, finditer
import sys
from sys import maxsize, stdin
import json
from json import loads
import math
from math import floor, ceil, factorial, sqrt, isqrt, inf, log2, log10, sin, cos, tan, pi, e, comb, perm, gcd, lcm
import copy
import pickle
import heapq
from heapq import heappush, heappop, heapify, heappushpop, nlargest, nsmallest
import bisect
from bisect import bisect_left, bisect_right
import string
from string import ascii_letters, ascii_lowercase, ascii_uppercase, digits, whitespace, punctuation, hexdigits
import random
import operator
import itertools
from itertools import combinations, permutations, product, groupby, chain, accumulate, zip_longest
import functools
from functools import lru_cache, cache, reduce
import collections
from collections import OrderedDict, defaultdict, Counter, deque
from typing import Set, Dict, List, Optional, Tuple

import sortedcontainers # pip install sortedcontainers
from sortedcontainers import SortedList, SortedDict, SortedSet
""",
    "java": r"""
import java.io.*;
import java.math.*;
import java.text.*;
import java.util.*;
import java.util.stream.*;
import java.util.function.*;
""",
    "javascript": r"""
const util = require('util');
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const assert = require('assert');
const os = require('os');
const http = require('http');
const https = require('https');
const url = require('url');
const querystring = require('querystring');
const zlib = require('zlib');
const stream = require('stream');
const buffer = require('buffer');
const events = require('events');
const child_process = require('child_process');
const readline = require('readline');
const process = require('process');
const string_decoder = require('string_decoder');
const timers = require('timers');
const perf_hooks = require('perf_hooks');
const dgram = require('dgram');  // UDP
const dns = require('dns');
const net = require('net');  // TCP
const tls = require('tls');
const vm = require('vm');

const _ = require('lodash');
const { PriorityQueue, Queue, Deque } = require('datastructures-js');
""",
    "cpp": r"""
// #include <bits/stdc++.h>
#include <iostream>
#include <algorithm>
#include <array>
#include <bitset>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>
#include <deque>
#include <forward_list>
#include <functional>
#include <iomanip>
#include <map>
#include <numeric>
#include <queue>
#include <set>
#include <stack>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>
#include <list>
#include <bitset>
#include <fstream>
#include <sstream>
#include <iterator>
#include <random>
#include <chrono>
#include <memory>
#include <thread>
#include <mutex>
#include <utility>
#include <climits>
#include <tuple>
#include <cstdlib>
#include <cctype>

using namespace std;
""",
    "golang": r"""
package main

import (
	"io"
	"os"
	"fmt"
	"math"
	"sort"
	"time"
	"bufio"
	"regexp"
	"reflect"
	"strings"
	"strconv"
	"math/big"
	"math/bits"
	"math/rand"
	"container/heap"
	"container/list"
)
""",
    "ruby": r"""
require 'set'
require 'date'
require 'time'
require 'stringio'
require 'bigdecimal'
require 'securerandom'

require 'json'
require 'algorithms'
""",
    "rust": r"""
use std::mem;
use std::fmt::{self, Display};
use std::iter::FromIterator;
use std::io::{self, Read, Write};
use std::cmp::{min, max, Ordering};
use std::collections::{HashMap, HashSet, BTreeMap, BTreeSet, VecDeque, BinaryHeap};
"""
}

EFFIBENCH_REGISTRY = {
    "cpp": {
        "id": 0,
        "verbose_name": "C++",
        "md_langs": ["cpp", "c++"],
        "llm_sandbox_lang": "cpp",
        "packages": [],
        "imports": _LANGUAGE_IMPORTS["cpp"],
        "flags": ["-O2", "-fsanitize=address"],
        "image": "gcc:14.2.0-bookworm",
        "executable": "g++",
    },
    "java": {
        "id": 1,
        "verbose_name": "Java",
        "md_langs": ["java"],
        "llm_sandbox_lang": "java",
        "packages": [],
        "imports": _LANGUAGE_IMPORTS["java"],
        "image": "openjdk:21-jdk-bookworm",
        "executable": "java",
    },
    "javascript": {
        "id": 6,
        "verbose_name": "JavaScript",
        "md_langs": ["js", "javascript", "node"],
        "llm_sandbox_lang": "javascript",
        "packages": ["lodash", "datastructures-js"],
        "imports": _LANGUAGE_IMPORTS["javascript"],
        "flags": ["--harmony"],
        "image": "node:22.14.0-bookworm",
        "executable": "node",
    },
    "ruby": {
        "id": 7,
        "verbose_name": "Ruby",
        "md_langs": ["rb", "ruby", "jruby", "macruby", "rake", "rbx"],
        "llm_sandbox_lang": "ruby",
        "packages": ["json", "algorithms"],
        "imports": _LANGUAGE_IMPORTS["ruby"],
        "flags": [],
        "image": "ruby:3.2.7-bookworm",
        "executable": "ruby",
    },
    "golang": {
        "id": 10,
        "verbose_name": "Go",
        "md_langs": ["golang", "go"],
        "llm_sandbox_lang": "go",
        "packages": [],
        "imports": _LANGUAGE_IMPORTS["golang"],
        "flags": [],
        "image": "golang:1.23.7-bookworm",
        "executable": "go",
    },
    "python3": {
        "id": 11,
        "verbose_name": "Python3",
        "md_langs": ["python", "py"],
        "llm_sandbox_lang": "python",
        "packages": ["sortedcontainers"],
        "imports": _LANGUAGE_IMPORTS["python3"],
        "flags": [],
        "image": "python:3.11.11-bookworm",
        "executable": "python3",
    },
}

EFFIBENCH_LANGS = list(EFFIBENCH_REGISTRY.keys())

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

def get_token(provider_name: str) -> str | None:
    """
    Returns the token for the specified provider name from TOKEN_REGISTRY.
    Returns None if not found.
    """
    return TOKEN_REGISTRY.get(provider_name)

def get_provider_name(provider_name: str) -> str | None:
    """
    Returns the provider name for the specified token from TOKEN_REGISTRY.
    Returns None if not found.
    """
    if provider_name == "local":
        return None
    else:
        return provider_name

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

def get_md_lang(lang: str) -> str | None:
    """
    Returns the first Markdown code block identifier for the given language key from LANG_LOOKUP.
    Returns None if not found.
    """
    md_langs = LANGUAGE_REGISTRY.get(lang, {}).get("md_langs", [])
    return md_langs[0] if md_langs else None

def get_lang_by_md_lang(md_lang: str) -> str | None:
    """
    Returns the language key for the given Markdown code block identifier from LANG_LOOKUP.
    Returns None if not found.
    """
    if md_lang in LANGUAGE_REGISTRY["python3"]["md_langs"]:
        return "python3"
    return next((key for key, value in LANGUAGE_REGISTRY.items() if md_lang in value["md_langs"]), None)

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
    if inference_provider == "openai":
        openai_token = get_token(inference_provider)
        if openai_token is None:
            raise ValueError(f"No token found for provider: {inference_provider}")
        client = OpenAI(api_key=openai_token)
        response = client.responses.create(
            model=model_name,
            input=[{
                "role": "user", 
                "content": [{"type": "input_text", "text": prompt}]
            }],
            text={"format": {"type": "text"}},
        )
        return response.output_text
    elif inference_provider == "claude":
        anthropic_token = get_token(inference_provider)
        if anthropic_token is None:
            raise ValueError(f"No token found for provider: {inference_provider}")
        client = anthropic.Anthropic(api_key=anthropic_token)
        response = client.messages.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.content[0].text
    
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

def postprocess_test_runner(test_runner: str, lang: str) -> str:
    # Prune import statements from the test runner
    test_runner = prune_package_imports(test_runner, lang)
    # Regularize the code submission placeholder
    lines = test_runner.split("\n")
    for i, line in enumerate(lines):
        if "==Code Submission==" in line:
            lines[i] = "==Code Submission=="
    test_runner = "\n".join(lines)
    return test_runner

def prune_package_imports(code: str, lang: str) -> str:
    """
    Prunes all import statements from code and additionally 'package main' for Go.
    
    Args:
        code: Source code string to process
        lang: Programming language identifier (e.g., "python3", "golang")
        
    Returns:
        Code string with all imports removed
        
    Raises:
        ValueError: If language is not supported
    """
    if lang not in EFFIBENCH_REGISTRY:
        raise ValueError(f"Language '{lang}' is not supported. Supported languages: {EFFIBENCH_LANGS}")

    # Handle empty code
    if not code.strip():
        return ""

    # Define language-specific import patterns and special processing rules
    patterns = {
        "python3": [
            r'^\s*import\s+.*$',
            r'^\s*from\s+\w+.*\s+import.*$'
        ],
        "javascript": [
            r'^\s*import\s+.*$',
            r'^\s*(const|let|var)\s+.*?=\s*require\(.*$',
            r'^\s*require\(.*$'
        ],
        "java": [
            r'^\s*import\s+.*$'
        ],
        "cpp": [
            r'^\s*#include\s+.*$',
            r'^\s*using\s+namespace\s+.*$',
            r'^\s*using\s+\w+::.*$'
        ],
        "ruby": [
            r'^\s*(require|require_relative)\s+.*$'
        ],
        "golang": [
            r'^\s*import\s+.*$',
            r'^\s*package\s+main\s*$'
        ]
    }

    lines = code.splitlines()
    result_lines = []

    # Special processing for multi-line imports and blocks
    skip_until_pattern = None  # Use this to skip lines in multi-line blocks
    skip_current_line = False
    custom_go_package = None   # Store Go custom package if found

    # First pass: process all lines
    for i, line in enumerate(lines):
        skip_current_line = False

        # Handle multi-line skipping (for parenthesized imports)
        if skip_until_pattern:
            if re.search(skip_until_pattern, line):
                skip_until_pattern = None
            continue

        # Language-specific special processing
        if lang == "python3":
            # Handle Python's multi-line imports with parentheses
            if re.match(r'^\s*from\s+\w+.*\s+import\s+\(', line) and ')' not in line:
                skip_until_pattern = r'\)'
                skip_current_line = True

        elif lang == "golang":
            # Handle Go's custom package and import blocks
            if re.match(r'^\s*package\s+([a-zA-Z0-9_]+)\s*$', line):
                match = re.match(r'^\s*package\s+([a-zA-Z0-9_]+)\s*$', line)
                if match and match.group(1) != "main":
                    custom_go_package = line
                skip_current_line = True
            
            if re.match(r'^\s*import\s*\($', line) or line.strip() == "import (":
                skip_until_pattern = r'^\s*\)\s*$'
                skip_current_line = True

        # Check against language-specific patterns
        for pattern in patterns.get(lang, []):
            if re.match(pattern, line):
                skip_current_line = True
                break

        # Special case for JavaScript comments (preserve them)
        if lang == "javascript" and re.match(r'^\s*(//|/\*)', line):
            result_lines.append(line)
            continue

        # Add non-import lines to result
        if not skip_current_line:
            result_lines.append(line)

    # Add Go custom package if found
    if lang == "golang" and custom_go_package:
        result_lines.insert(0, custom_go_package)

    # Join the pruned lines
    pruned_code = '\n'.join(result_lines)

    # Clean up blank lines
    pruned_code = re.sub(r'\n{3,}', '\n\n', pruned_code)  # No more than 2 consecutive newlines
    pruned_code = pruned_code.strip()

    return pruned_code

def postprocess_solution(solution: str, lang: str) -> str:
    # Prune import statements from the solution
    solution = prune_package_imports(solution, lang)
    return solution

def postprocess_full_code_java(test_runner: str) -> str:
    # Find the main class with public static void main method
    main_class_match = re.search(r'public\s+class\s+(\w+).*?public\s+static\s+void\s+main', test_runner, re.DOTALL)
    if main_class_match and test_runner.count("public class") > 1:
        main_class = main_class_match.group(1)
        # Replace all other "public class X" with "class X"
        test_runner = re.sub(
            r'public\s+class\s+(?!%s\b)(\w+)' % re.escape(main_class),
            r'class \1',
            test_runner
        )
    return test_runner

def get_full_code(lang: str, solution: str, test_runner: str) -> str:
    test_runner = postprocess_test_runner(test_runner, lang)
    solution = postprocess_solution(solution, lang)
    code = EFFIBENCH_REGISTRY.get(lang, {}).get('imports', '') + "\n\n" + test_runner.replace("==Code Submission==", solution)
    return postprocess_full_code_java(code)
