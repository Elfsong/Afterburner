import sys
from io import StringIO

# 假设你的代码需要两次输入
def my_program():
    name = input("Enter your name: ")
    age = input("Enter your age: ")
    print(f"Name: {name}, Age: {age}")

# 模拟输入
mock_input = StringIO("Mingzhe\n25\n")
sys.stdin = mock_input

my_program()

# 恢复 stdin
sys.stdin = sys.__stdin__
