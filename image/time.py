import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(10, 6)) # (width, height) in inches
iterations = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) # X-axis values: Number of iterations

# Data for "Base Model" - Time%
percent = np.array([12.04, 12.74, 13.08, 13.25, 13.42, 13.50, 13.59, 13.68, 13.69, 13.69, 13.71]) 
ax.plot(iterations, percent, marker='o', linestyle='-', color='gray', label='Base Model')
for i, txt in enumerate(percent):
    ax.annotate(f'{txt:.2f}%', (iterations[i], percent[i]), textcoords="offset points", xytext=(0,10), ha='center')

# Data for "Afterburner 3B-SFT" - Time%
percent = np.array([25.04, 26.00, 26.40, 26.55, 26.70, 26.72, 26.74, 26.76, 26.78, 26.78, 26.78])
ax.plot(iterations, percent, marker='o', linestyle='-', color='darkorange', label='Afterburner SFT')
for i, txt in enumerate(percent):
    ax.annotate(f'{txt:.2f}%', (iterations[i], percent[i]), textcoords="offset points", xytext=(0,10), ha='center')

# Data for "Afterburner 3B-DPO" - Time%
percent = np.array([24.61, 26.43, 27.93, 28.03, 28.47, 29.04, 29.46, 29.90, 30.42, 30.90, 30.92]) 
ax.plot(iterations, percent, marker='o', linestyle='-', color='darkblue', label='Afterburner DPO')
for i, txt in enumerate(percent):
    ax.annotate(f'{txt:.2f}%', (iterations[i], percent[i]), textcoords="offset points", xytext=(0,10), ha='center')

# Data for "Afterburner 3B-GRPO" - Time%
percent = np.array([31.22, 36.69, 38.41, 39.64, 40.83, 42.35, 43.49, 44.30, 44.82, 45.17, 45.17])
ax.plot(iterations, percent, marker='o', linestyle='-', color='forestgreen', label='Afterburner GRPO')
for i, txt in enumerate(percent):
    ax.annotate(f'{txt:.2f}%', (iterations[i], percent[i]), textcoords="offset points", xytext=(0,10), ha='center')


ax.set_title('Time% v.s. Iterations', fontsize=16)
ax.set_xlabel('Iteration', fontsize=14)
ax.set_ylabel('Time%', fontsize=14)
ax.set_xticks(iterations)
ax.legend(fontsize=12)
ax.grid(True, linestyle='--', alpha=0.7)

plt.style.use('seaborn-v0_8-whitegrid')
plt.tight_layout()
plt.show()
plt.savefig('time.png', dpi=600)