
from jitter import Jitter

jitter = Jitter(number_of_cases=5, timeout=30)

# code = """
# class Solution:
#     def rotate(self, matrix: List[List[int]]) -> None:
#         n=len(matrix)
#         for i in range(n):
#             for j in range(i):
#                 matrix[i][j],matrix[j][i]=matrix[j][i],matrix[i][j]
#         for i in range(n):
#             matrix[i].reverse()
# """

code = """
class Solution:
    def getCoprimes(self, nums: List[int], edges: List[List[int]]) -> List[int]:
        ans = [-1] * len(nums)
        path = [[] for _ in range(51)]
        graph = defaultdict(list)
        seen = set()
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)

        def dfs(node, depth):
            if node in seen: return
            seen.add(node)
            largestDepth = -1
            for x in range(1, 51):
                if gcd(nums[node], x) == 1: # co-prime
                    if len(path[x]) > 0:
                        topNode, topDepth = path[x][-1]
                        if largestDepth < topDepth:  # Pick node which has largestDepth and co-prime with current node as our ancestor node
                            largestDepth = topDepth
                            ans[node] = topNode
            path[nums[node]].append((node, depth))
            for nei in graph[node]:
                dfs(nei, depth + 1)
            path[nums[node]].pop()

        dfs(0, 0)
        return ans
"""

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