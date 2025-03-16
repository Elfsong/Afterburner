
from jitter import Jitter

code = """
class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        n=len(matrix)
        for i in range(n):
            for j in range(i):
                matrix[i][j],matrix[j][i]=matrix[j][i],matrix[i][j]
        for i in range(n):
            matrix[i].reverse()
"""
jitter = Jitter(number_of_cases=5, timeout=30)
test_cases, test_case_generator_code, test_case_construction = jitter.generate(code)

print("======== Test Cases ========")
print(test_cases)

print("======== Test Case Generator Code ========")
print(test_case_generator_code)

print("======== Libraries ========")
print(test_case_construction['libraries'])

print("======== Import Statements ========")
print(test_case_construction['import_statements'])

print("======== Executable Code ========")
print(test_case_construction['executable_code']) 
