# coding: utf-8

# Author: Du Mingzhe (mingzhe@nus.edu.sg)
# Date: 2025-03-18

import requests

class Monolith:
    def __init__(self, backend_url='https://monolith.cool') -> None:
        self.backend_url = backend_url
        
    def post_code_submit(self, lang, libs, code: str, timeout: int, profiling: bool) -> str:
        data = {
            'code': code,
            'language': lang,
            'libraries': libs,
            'timeout': timeout,
            'run_memory_profile': profiling
        }

        response = requests.post(f'{self.backend_url}/execute', json=data)
        return response.json()

    def get_code_result(self, task_id: str) -> str:
        response = requests.get(f'{self.backend_url}/results/{task_id}')
        return response.json()