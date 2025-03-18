It's Monolith, a code execution environment. You can run code in a variety of languages (Python / Golang / Cpp / Java / Javascript), and see the output.

Should you have any questions, please don't hesitate to ask mingzhe@nus.edu.sg.

# Quickstart
### Installation
```bash
pip install monolith-lib
```

### Function Calls
```python
from monolith import monolith

monolith = monolith.Monolith(backend_url='https://monolith.cool')

# 1) Submit code to Monolith (POST)
post_response = monolith.post_code_submit(
    lang = 'python',
    libs = [],
    code = 'print("Hello, World!")',
    timeout = 10,
    profiling = False
)

# 2) Get async task_id from POST response
task_id = post_response['task_id']

# 3) Get code result from Monolith (GET)
get_response = monolith.get_code_result(task_id)
print(get_response)
```