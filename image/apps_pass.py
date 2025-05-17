import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(10, 6)) # (width, height) in inches
iterations = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) # X-axis values: Number of iterations

# Data for "Base Model" - Pass@1
percent = np.array([10.67, 11.67, 13.00, 13.67, 15.00, 15.00, 15.00, 15.33, 15.67, 15.67, 15.67]) # Y-axis values: Mean Integral Percentage
ax.plot(iterations, percent, marker='o', linestyle='-', color='gray', label='Base Model')
for i, txt in enumerate(percent):
    ax.annotate(f'{txt:.2f}%', (iterations[i], percent[i]), textcoords="offset points", xytext=(0,10), ha='center')

# Data for "Afterburner 3B-SFT" - Pass@1
percent = np.array([15.67, 18.00, 18.00, 18.67, 19.00, 20.33, 21.00, 21.00, 21.00, 21.33, 21.33]) # Y-axis values: Mean Integral Percentage
ax.plot(iterations, percent, marker='o', linestyle='-', color='darkorange', label='Afterburner SFT')
for i, txt in enumerate(percent):
    ax.annotate(f'{txt:.2f}%', (iterations[i], percent[i]), textcoords="offset points", xytext=(0,10), ha='center')

# Data for "Afterburner 3B-DPO" - Pass@1
percent = np.array([17.00, 20.33, 20.33, 20.33, 21.00, 21.00, 21.67, 22.33, 22.67, 22.67, 23.00]) # Y-axis values: Mean Integral Percentage
ax.plot(iterations, percent, marker='o', linestyle='-', color='darkblue', label='Afterburner DPO')
for i, txt in enumerate(percent):
    ax.annotate(f'{txt:.2f}%', (iterations[i], percent[i]), textcoords="offset points", xytext=(0,10), ha='center')

# Data for "Afterburner 3B-GRPO" - Pass@1
percent = np.array([13.00, 15.00, 16.67, 20.33, 21.67, 24.00, 26.67, 28.00, 30.00, 31.33, 31.67]) # Y-axis values: Mean Integral Percentage
ax.plot(iterations, percent, marker='o', linestyle='-', color='forestgreen', label='Afterburner GRPO')
for i, txt in enumerate(percent):
    ax.annotate(f'{txt:.2f}%', (iterations[i], percent[i]), textcoords="offset points", xytext=(0,10), ha='center')


ax.set_title('Pass@1 v.s. Iterations', fontsize=16)
ax.set_xlabel('Iteration', fontsize=14)
ax.set_ylabel('Pass@1', fontsize=14)
ax.set_xticks(iterations)
ax.legend(fontsize=12)
ax.grid(True, linestyle='--', alpha=0.7)

plt.style.use('seaborn-v0_8-whitegrid')
plt.tight_layout()
plt.show()
plt.savefig('apps_pass.png', dpi=600)