# coding: utf-8

# Author: Du Mingzhe (mingzhe@nus.edu.sg)
# Date: 2025-03-20

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class Monolith:
    def __init__(self, backend_url='https://monolith.cool', retries=5, backoff_factor=0.2) -> None:
        self.backend_url = backend_url
        self.session = requests.Session()
        self.status_forcelist = (502, 503, 504)
        
        retry_strategy = Retry(
            total=retries,
            backoff_factor=backoff_factor,
            status_forcelist=self.status_forcelist,
            allowed_methods=["GET", "POST"]
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
            
    def post_code_submit(self, lang, libs, code: str, timeout: int, profiling: bool) -> dict:
        data = {
            'code': code,
            'language': lang,
            'libraries': libs,
            'timeout': timeout,
            'run_memory_profile': profiling
        }
        
        response_dict = {'task_id': None, 'status': 'error', 'error': 'unknown'}
        try:
            response = self.session.post(f'{self.backend_url}/execute', json=data, timeout=timeout)
            if response.status_code == 200:
                response = response.json()
                response_dict['task_id'] = response.get('task_id', None)
                response_dict['status'] = response.get('status', None)
                response_dict['error'] = response.get('error', None)
            else:
                raise Exception(f"Error {response.status_code}: {response.text}")
        except Exception as e:
            response_dict['task_id'] = None
            response_dict['status'] = 'error'
            response_dict['error'] = str(e)
        finally:
            return response_dict

    def get_code_result(self, task_id: str) -> dict:
        response = self.session.get(f'{self.backend_url}/results/{task_id}')
        return response.json()
    
    def get_status(self) -> dict:
        response = self.session.get(f'{self.backend_url}/status')
        return response.json()
    
    def clean(self, result_limit=0) -> dict:
        response = self.session.delete(f'{self.backend_url}/clean/{result_limit}')
        return response.json()
    
# python setup.py sdist
# twine upload dist/*
