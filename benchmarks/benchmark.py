# coding: utf-8

# Author: Du Mingzhe (mingzhe@nus.edu.sg)
# Date: 2025-03-27

import os
import logging
from typing import Any
from datasets import load_dataset

# Logging Configuration
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
logger.addHandler(handler)

class Benchmark:
    def __init__(self, dataset_name) -> None:
        self.dataset_name = dataset_name
        self.logger = logger.getChild(self.dataset_name)
        self.logger.info(f"[+] Initialized Benchmark for dataset [{dataset_name}].")
        
    def inference(self, instance) -> Any:
        raise NotImplementedError("Method 'inference' must be implemented.")
    
    def instance_eval(self, instance) -> Any:
        raise NotImplementedError("Method 'instance_eval' must be implemented.")
    
    def eval(self) -> Any:
        raise NotImplementedError("Method 'eval' must be implemented.")
    
class APPSBenchmark(Benchmark):
    def __init__(self, dataset_name) -> None:
        super().__init__(dataset_name)
        
        # Load the dataset
        self.logger.info(f"[+] Loading dataset [{self.dataset_name}]...")
        self.ds = load_dataset("Elfsong/apps_verified", "default")
        
    def inference(self, instance) -> Any:
        return instance
    
    def instance_eval(self, instance) -> Any:
        return instance
    
    def eval(self) -> Any:
        return 0.0
    
class MercuryBenchmark(Benchmark):
    def __init__(self, dataset_name) -> None:
        super().__init__(dataset_name)
        
        # Load the dataset
        self.logger.info(f"[+] Loading dataset [{self.dataset_name}]...")
        self.ds = load_dataset("Elfsong/Mercury", split="eval")
        
    def inference(self, instance) -> Any:
        return instance
    
    def instance_eval(self, instance) -> Any:
        return instance
    
    def eval(self) -> Any:
        return 0.0
    
if __name__ == '__main__':
    apps_benchmark = APPSBenchmark("apps_verified")
    mercury_benchmark = MercuryBenchmark("mercury")