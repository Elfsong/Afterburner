import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(10, 6)) # (width, height) in inches
iterations = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) # X-axis values: Number of iterations

# Data for "Base Model" - Pass@1
percent = np.array([27.99, 28.33, 28.67, 28.67, 29.00, 29.08, 29.17, 29.24, 29.33, 29.33, 29.33]) # Y-axis values: Mean Integral Percentage
ax.plot(iterations, percent, marker='o', linestyle='-', color='gray', label='Base Model')
for i, txt in enumerate(percent):
    ax.annotate(f'{txt:.2f}%', (iterations[i], percent[i]), textcoords="offset points", xytext=(0,10), ha='center')

# Data for "Afterburner 3B-SFT" - Pass@1
percent = np.array([46.00, 46.00, 46.33, 47.00, 48.33, 48.33, 48.67, 48.67, 48.67, 48.67, 48.67]) # Y-axis values: Mean Integral Percentage
ax.plot(iterations, percent, marker='o', linestyle='-', color='darkorange', label='Afterburner SFT')
for i, txt in enumerate(percent):
    ax.annotate(f'{txt:.2f}%', (iterations[i], percent[i]), textcoords="offset points", xytext=(0,10), ha='center')

# Data for "Afterburner 3B-DPO" - Pass@1
percent = np.array([43.00, 50.00, 51.33, 51.50, 51.67, 51.67, 51.67, 51.67, 51.67, 51.67, 51.67]) # Y-axis values: Mean Integral Percentage
ax.plot(iterations, percent, marker='o', linestyle='-', color='darkblue', label='Afterburner DPO')
for i, txt in enumerate(percent):
    ax.annotate(f'{txt:.2f}%', (iterations[i], percent[i]), textcoords="offset points", xytext=(0,10), ha='center')

# Data for "Afterburner 3B-GRPO" - Pass@1
percent = np.array([37.33, 48.33, 52.00, 54.50, 57.00, 58.17, 59.34, 60.50, 61.18, 61.67, 61.67]) # Y-axis values: Mean Integral Percentage
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
plt.savefig('pass.png', dpi=600)