# coding: utf-8

# Author: Du Mingzhe (mingzhe@nus.edu.sg)
# Date: 2025-03-27

import os
from typing import Any

class Benchmark:
    def __init__(self, dataset_name) -> None:
        self.dataset_name = dataset_name
        
    def inference(self, instance) -> Any:
        raise NotImplementedError("Method 'inference' must be implemented.")
    
    def instance_eval(self, instance) -> Any:
        raise NotImplementedError("Method 'instance_eval' must be implemented.")
    
    def eval(self) -> Any:
        raise NotImplementedError("Method 'eval' must be implemented.")
    
class APPSBenchmark(Benchmark):
    def __init__(self, dataset_name) -> None:
        super().__init__(dataset_name)
        
    def inference(self, instance) -> Any:
        return instance
    
    def instance_eval(self, instance) -> Any:
        return instance
    
    def eval(self) -> Any:
        return 0.0
    
class MercuryBenchmark(Benchmark):
    def __init__(self, dataset_name) -> None:
        super().__init__(dataset_name)
        
    def inference(self, instance) -> Any:
        return instance
    
    def instance_eval(self, instance) -> Any:
        return instance
    
    def eval(self) -> Any:
        return 0.0