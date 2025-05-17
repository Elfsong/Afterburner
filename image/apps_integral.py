import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(10, 6)) # (width, height) in inches
iterations = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) # X-axis values: Number of iterations

# Data for "Base Model" - Integral%
integral_percent = np.array([4.14, 4.14, 4.14, 4.14, 4.23, 4.23, 4.24, 4.53, 4.53, 4.53, 4.53])
ax.plot(iterations, integral_percent, marker='o', linestyle='-', color='gray', label='Base Model')
for i, txt in enumerate(integral_percent):
    ax.annotate(f'{txt:.2f}%', (iterations[i], integral_percent[i]), textcoords="offset points", xytext=(0,10), ha='center')

# Data for "Afterburner 3B-SFT" - Integral%
integral_percent = np.array([6.93, 8.15, 8.15, 8.27, 8.44, 8.48, 8.83, 9.01, 9.01, 9.01, 9.05])
ax.plot(iterations, integral_percent, marker='o', linestyle='-', color='darkorange', label='Afterburner SFT')
for i, txt in enumerate(integral_percent):
    ax.annotate(f'{txt:.2f}%', (iterations[i], integral_percent[i]), textcoords="offset points", xytext=(0,10), ha='center')

# Data for "Afterburner 3B-DPO" - Integral%
integral_percent = np.array([8.49, 10.36, 10.57, 10.73, 11.03, 11.03, 11.11, 11.27, 11.27, 11.27, 11.27])
ax.plot(iterations, integral_percent, marker='o', linestyle='-', color='darkblue', label='Afterburner DPO')
for i, txt in enumerate(integral_percent):
    ax.annotate(f'{txt:.2f}%', (iterations[i], integral_percent[i]), textcoords="offset points", xytext=(0,10), ha='center')

# Data for "Afterburner 3B-GRPO" - Integral%
integral_percent = np.array([4.69, 9.04, 11.42, 12.36, 14.01, 14.85, 15.22, 16.01, 16.18, 16.18, 16.18])
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
plt.savefig('apps_integral.png', dpi=600)