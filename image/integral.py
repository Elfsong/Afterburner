import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(10, 6)) # (width, height) in inches
iterations = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) # X-axis values: Number of iterations

# Data for "Base Model" - Integral%
integral_percent = np.array([10.29, 10.77, 11.25, 11.49, 11.72, 11.83, 11.96, 12.07, 12.19, 12.19, 12.19])
ax.plot(iterations, integral_percent, marker='o', linestyle='-', color='gray', label='Base Model')
for i, txt in enumerate(integral_percent):
    ax.annotate(f'{txt:.2f}%', (iterations[i], integral_percent[i]), textcoords="offset points", xytext=(0,10), ha='center')

# Data for "Afterburner 3B-SFT" - Integral%
integral_percent = np.array([21.01, 21.09, 21.50, 21.88, 22.25, 22.31, 22.38, 22.44, 22.50, 22.50, 22.50])
ax.plot(iterations, integral_percent, marker='o', linestyle='-', color='darkorange', label='Afterburner SFT')
for i, txt in enumerate(integral_percent):
    ax.annotate(f'{txt:.2f}%', (iterations[i], integral_percent[i]), textcoords="offset points", xytext=(0,10), ha='center')

# Data for "Afterburner 3B-DPO" - Integral%
integral_percent = np.array([19.13, 26.25, 27.05, 27.42, 27.95, 28.11, 28.51, 29.51, 29.51, 29.51, 29.51])
ax.plot(iterations, integral_percent, marker='o', linestyle='-', color='darkblue', label='Afterburner DPO')
for i, txt in enumerate(integral_percent):
    ax.annotate(f'{txt:.2f}%', (iterations[i], integral_percent[i]), textcoords="offset points", xytext=(0,10), ha='center')

# Data for "Afterburner 3B-GRPO" - Integral%
integral_percent = np.array([16.24, 24.81, 29.44, 30.85, 33.56, 35.48, 37.09, 38.01, 38.62, 38.95, 38.96])
ax.plot(iterations, integral_percent, marker='o', linestyle='-', color='forestgreen', label='Afterburner GRPO')
for i, txt in enumerate(integral_percent):
    ax.annotate(f'{txt:.2f}%', (iterations[i], integral_percent[i]), textcoords="offset points", xytext=(0,10), ha='center')


ax.set_title('Integral% v.s. Iterations', fontsize=16)
ax.set_xlabel('Iteration', fontsize=14)
ax.set_ylabel('Integral%', fontsize=14)
ax.set_xticks(iterations)
ax.legend(fontsize=12)
ax.grid(True, linestyle='--', alpha=0.7)

plt.style.use('seaborn-v0_8-whitegrid')
plt.tight_layout()
plt.show()
plt.savefig('integral.png', dpi=600)