
from jitt import JITT

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
jitt = JITT(number_of_cases=5, timeout=30)
response_dict = jitt.generate(code)

print("======== Test Cases ========")
print(response_dict['test_cases'])

print("======== Test Case Generator Code ========")
print(response_dict['test_case_generator_code'])

print("======== Libraries ========")
print(response_dict['test_case_construction']['libraries'])

print("======== Import Statements ========")
print(response_dict['test_case_construction']['import_statements'])

print("======== Executable Code ========")
print(response_dict['test_case_construction']['executable_code']) 
