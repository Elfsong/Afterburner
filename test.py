import io
import sys
import unittest
from typing import List

def solution():
    class Solution:
        def twoSum(self, nums: List[int], target: int) -> List[int]:
            numMap = {}
            n = len(nums)

            # Build the hash table
            for i in range(n):
                numMap[nums[i]] = i

            # Find the complement
            for i in range(n):
                complement = target - nums[i]
                if complement in numMap and numMap[complement] != i:
                    return [i, numMap[complement]]

            return []  # No solution found


    import sys
    data = sys.stdin.read().strip()
    if data:
        arr_str, target_str = data.split(' ', 1)
        nums = list(map(int, arr_str.split(',')))
        target = int(target_str)
        sol = Solution()
        ans = sol.twoSum(nums, target)
        print(f"{ans[0]},{ans[1]}")

class TestSolution(unittest.TestCase):
    def run_io_fun(self, input_data):
        backup_stdin = sys.stdin
        backup_stdout = sys.stdout
        try:
            sys.stdin = io.StringIO(input_data)
            output_catcher = io.StringIO()
            sys.stdout = output_catcher

            solution()

            output_catcher.seek(0)
            return output_catcher.read()
        finally:
            sys.stdin = backup_stdin
            sys.stdout = backup_stdout

def make_test_function(input_data, expected):
    def test_function(self):
        actual = self.run_io_fun(input_data).strip()
        self.assertEqual(expected, actual)
    return test_function

test_case_list = [
    {'input': '3,3 6', 'output': '0,1'},
    {'input': '2,7,11,15 9', 'output': '0,1'},
    {'input': '3,2,4 6', 'output': '1,2'},
    {'input': '-10,20,10,5 10', 'output': '0,1'},
    {'input': '1,2,3,4,5 9', 'output': '3,4'}
]

for i, case in enumerate(test_case_list, start=1):
    test_name = f"test_case_{{i}}"
    test_func = make_test_function(case['input'], case['output'])
    setattr(TestSolution, test_name, test_func)

if __name__ == '__main__':
    result = unittest.main(verbosity=2, exit=False)
    
    # If all tests passed, print "Success".
    if result.result.wasSuccessful():
        print("Success")
    else:
        print("Failed")