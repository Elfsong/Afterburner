# coding: utf-8

# Author: Du Mingzhe (mingzhe@nus.edu.sg)
# Date: 2025-01-13

import json
import math
import utils
import random
from tqdm import tqdm
from client import Client
from datasets import load_dataset
from llm_sandbox import SandboxSession
from pydantic import BaseModel, ConfigDict

class CodeGeneration(BaseModel):
    model_config = ConfigDict(extra='forbid')  # required for openai
    draft: str
    code: str

class MCTSNode:
    def __init__(self, code: str, draft: str, parent=None):
        self.code = code
        self.draft = draft
        self.parent = parent
        self.children = []
        self.values = {"V": 0, "N": 0}
    
    def ucb1(self, exploration_constant=math.sqrt(2)):
        """Calculates the UCB1 value for this node."""
        if self.visit_count == 0:
            return float('inf')  # Encourage exploration of unvisited nodes
        exploitation = self.total_reward / self.visit_count
        exploration = exploration_constant * math.sqrt(math.log(self.parent.visit_count) / self.visit_count)
        return exploitation + exploration
    
    def select_child(self):
        """Selects the best child node based on UCB1."""
        best_child = None
        best_ucb1 = -float('inf')

        for action, child in self.children.items():
            ucb1_value = child.ucb1()
            if ucb1_value > best_ucb1:
                best_ucb1 = ucb1_value
                best_child = child
        return best_child

class AfterBurner:
    def __init__(self):
        self.model = Client(model_name="gpt-4o", model_type="openai").model
    
    def thrust(self, code: str, instruction: str, method: str) -> str:
        print(f"Thrusting the code with [{method}]")
        if method == "temperature_sampling":
            speedup_code = self.temperature_sampling(code, instruction)
        elif method == "monte_carlo_tree_search":
            speedup_code = self.monte_carlo_tree_search(code, instruction)
        else:
            raise ValueError(f"Unsupported method: {method}")
        return speedup_code
    
    def solution_evaluation(self, instance, solution_code: str) -> str:
        entry_point = instance['entry_point']
        import_code = instance['import_code']
        test_case_generator = instance['test_case_generator']
        test_cases = instance['test_case'].split("\n")
        test_cases = [json.loads(test_case) for test_case in test_cases if test_case]
        execution_code = utils.code_evalution_construct(import_code, solution_code, test_case_generator, entry_point, test_cases)

        try:
            with SandboxSession(lang="python", verbose=False) as session:
                session.setup(libraries=instance['libraries'])
                response = utils.run_with_timeout(session.run, 60, code=execution_code, run_memory_profile=True)
                response['status'] = "success"
                return response
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    
    def temperature_sampling(self, instance: dict, instruction: str, target: str, solution: str, k: int, temperature: float) -> dict:
        solution_result_list = list()
        best_solution_result = None
        pass_k = False
        runtime_speedup_ratio = 0.0
        memory_shrink_ratio = 0.0
        integeral_ratio = 0.0

        # Performace evaluation of the original solution
        result = self.solution_evaluation(instance, solution)
        original_solution_result = {"draft": "", "code": solution, "result": result}
        solution_result_list.append(original_solution_result)
        if not original_solution_result['result']['status'] == "success": 
            return {
                "pass_k": pass_k, "runtime_speedup_ratio": runtime_speedup_ratio, 
                "memory_shrink_ratio": memory_shrink_ratio, "integeral_ratio": integeral_ratio, 
                "solution_result_list": solution_result_list
            }
        
        # Sampling k solutions using temperature sampling
        for _ in range(k):
            prompt = utils.solution_prompt_construct(instance, instruction, solution)
            response = self.model.json_generate(CodeGeneration, prompt, temperature=temperature)
            result = self.solution_evaluation(instance, response.code)
            solution_result = {"draft": response.draft, "code": response.code, "result": result}
            solution_result_list.append(solution_result)

            if result['status'] == "success":
                pass_k = True
                if not best_solution_result or solution_result['result']['output'][target] < best_solution_result['result']['output'][target]:
                    best_solution_result = solution_result

        if best_solution_result:
            runtime_speedup_ratio = original_solution_result['result']['output']['duration'] / best_solution_result['result']['output']['duration']
            memory_shrink_ratio = original_solution_result['result']['output']['peak_memory'] / best_solution_result['result']['output']['peak_memory']
            integeral_ratio = original_solution_result['result']['output']['integral'] / best_solution_result['result']['output']['integral']

        return {
            "pass_k": pass_k, "runtime_speedup_ratio": runtime_speedup_ratio, "memory_shrink_ratio": memory_shrink_ratio, 
            "integeral_ratio": integeral_ratio, "solution_result_list": solution_result_list
        }
        
    def beam_search(self, code: str, instruction: str) -> str:
        pass

    def monte_carlo_tree_search(self, code: str, instruction: str) -> str:
        pass


if __name__ == "__main__":
    afterburner = AfterBurner()
    task_ds = load_dataset("Elfsong/Venus_t", "python3", split="eval")

    for instance in tqdm(list(task_ds)):
        instance = utils.solution_filter(instance)
        
        # Pass (without oracle solution)
        solutions = instance['rt_list'] + instance['mm_list']
        solution = random.choice(solutions)
        result = afterburner.temperature_sampling(instance=instance, instruction="Generate an alternative solution.", target='duration', solution=solution, k=5, temperature=0.8)
        print(result['pass_k'], result['runtime_speedup_ratio'], result['memory_shrink_ratio'], result['integeral_ratio'])
        
        # Runtime
        solutions = instance['rt_list']
        solution = random.choice(solutions)
        result = afterburner.temperature_sampling(instance=instance, instruction="Generate a solution with faster runtime.", target='duration', solution=solution, k=5, temperature=0.8)
        print(result['pass_k'], result['runtime_speedup_ratio'], result['memory_shrink_ratio'], result['integeral_ratio'])
        
        # Memory
        solutions = instance['mm_list']
        solution = random.choice(solutions)
        result = afterburner.temperature_sampling(instance=instance, instruction="Generate a solution with faster runtime.", target='peak_memory', solution=solution, k=5, temperature=0.8)
        print(result['pass_k'], result['runtime_speedup_ratio'], result['memory_shrink_ratio'], result['integeral_ratio'])
        
        # Integral