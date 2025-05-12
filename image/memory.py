import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(10, 6)) # (width, height) in inches
iterations = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) # X-axis values: Number of iterations

# Data for "Base Model" - Memory%
percent = np.array([13.24, 13.73, 14.22, 14.46, 14.71, 14.84, 14.97, 15.10, 15.19, 15.19, 15.19])
ax.plot(iterations, percent, marker='o', linestyle='-', color='gray', label='Base Model')
for i, txt in enumerate(percent):
    ax.annotate(f'{txt:.2f}%', (iterations[i], percent[i]), textcoords="offset points", xytext=(0,10), ha='center')

# Data for "Afterburner 3B-SFT" - Memory%
percent = np.array([23.87, 23.80, 24.20, 24.55, 24.90, 25.00, 25.10, 25.19, 25.29, 25.30, 25.30])
ax.plot(iterations, percent, marker='o', linestyle='-', color='darkorange', label='Afterburner SFT')
for i, txt in enumerate(percent):
    ax.annotate(f'{txt:.2f}%', (iterations[i], percent[i]), textcoords="offset points", xytext=(0,10), ha='center')

# Data for "Afterburner 3B-DPO" - Memory%
percent = np.array([20.53, 24.29, 26.31, 26.74, 27.17, 27.40, 27.67, 27.68, 27.68, 27.68, 27.69])
ax.plot(iterations, percent, marker='o', linestyle='-', color='darkblue', label='Afterburner DPO')
for i, txt in enumerate(percent):
    ax.annotate(f'{txt:.2f}%', (iterations[i], percent[i]), textcoords="offset points", xytext=(0,10), ha='center')

# Data for "Afterburner 3B-GRPO" - Memory%
percent = np.array([25.14, 30.21, 34.17, 37.45, 40.72, 42.55, 44.39, 46.22, 47.65, 48.05, 48.05])
ax.plot(iterations, percent, marker='o', linestyle='-', color='forestgreen', label='Afterburner GRPO')
for i, txt in enumerate(percent):
    ax.annotate(f'{txt:.2f}%', (iterations[i], percent[i]), textcoords="offset points", xytext=(0,10), ha='center')


ax.set_title('Memory% v.s. Iterations', fontsize=16)
ax.set_xlabel('Iteration', fontsize=14)
ax.set_ylabel('Memory%', fontsize=14)
ax.set_xticks(iterations)
ax.legend(fontsize=12)
ax.grid(True, linestyle='--', alpha=0.7)

plt.style.use('seaborn-v0_8-whitegrid')
plt.tight_layout()
plt.show()
plt.savefig('memory.png', dpi=600)