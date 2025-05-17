import matplotlib.pyplot as plt
import numpy as np

# Overall figure and subplots
# Create a figure with 1 row and 4 columns of subplots
# Adjust figsize to accommodate all subplots; you might need to tweak this
fig, axs = plt.subplots(1, 2, figsize=(14, 3.5)) # Increased width and height slightly for better annotation spacing

plt.style.use('seaborn-v0_8-whitegrid') # Apply style once for the whole figure

iterations = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) # X-axis values: Number of iterations

# --- Plot 1: Pass@1 v.s. Iterations ---
ax = axs[0] # Select the first subplot

# Data for "Base Model" - Pass@1
percent = np.array([10.67, 11.67, 13.00, 13.67, 15.00, 15.00, 15.00, 15.33, 15.67, 15.67, 15.67]) # Y-axis values: Mean Integral Percentage
ax.plot(iterations, percent, marker='o', linestyle='-', color='gray', label='Base Model')
for i, txt in enumerate(percent):
    ax.annotate(f'{txt:.2f}%', (iterations[i], percent[i]), textcoords="offset points", xytext=(0,10), ha='center', color='gray', fontsize=7)

# Data for "Afterburner 3B-SFT" - Pass@1
percent = np.array([15.67, 18.00, 18.00, 18.67, 19.00, 20.33, 21.00, 21.00, 21.00, 21.33, 21.33]) # Y-axis values: Mean Integral Percentage
ax.plot(iterations, percent, marker='o', linestyle='-', color='darkorange', label='Afterburner SFT')
for i, txt in enumerate(percent):
    ax.annotate(f'{txt:.2f}%', (iterations[i], percent[i]), textcoords="offset points", xytext=(0,-12), ha='center', fontsize=7)

# Data for "Afterburner 3B-DPO" - Pass@1
percent = np.array([17.00, 20.33, 20.33, 20.33, 21.00, 21.00, 21.67, 22.33, 22.67, 22.67, 23.00]) # Y-axis values: Mean Integral Percentage
ax.plot(iterations, percent, marker='o', linestyle='-', color='darkblue', label='Afterburner DPO')
for i, txt in enumerate(percent):
    ax.annotate(f'{txt:.2f}%', (iterations[i], percent[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=7)

# Data for "Afterburner 3B-GRPO" - Pass@1
percent = np.array([13.00, 15.00, 16.67, 20.33, 21.67, 24.00, 26.67, 28.00, 30.00, 31.33, 31.67]) # Y-axis values: Mean Integral Percentage
ax.plot(iterations, percent, marker='o', linestyle='-', color='forestgreen', label='Afterburner GRPO')
for i, txt in enumerate(percent):
    ax.annotate(f'{txt:.2f}%', (iterations[i], percent[i]), textcoords="offset points", xytext=(0,10), ha='center', color='forestgreen', fontsize=7)

ax.set_title('Pass@1 v.s. Iterations', fontsize=14) # Adjusted fontsize
ax.set_ylabel('Pass@1 (%)', fontsize=12) # Added (%) for clarity, adjusted fontsize
ax.set_xticks(iterations)
ax.tick_params(axis='both', which='major', labelsize=10) # Adjusted fontsize
# Individual legend removed
ax.grid(True, linestyle='--', alpha=0.5)


# --- Plot 2: Time% v.s. Iterations ---
ax = axs[1] # Select the second subplot

# Data for "Base Model" - Integral%
integral_percent = np.array([4.14, 4.14, 4.14, 4.14, 4.23, 4.23, 4.24, 4.53, 4.53, 4.53, 4.53])
ax.plot(iterations, integral_percent, marker='o', linestyle='-', color='gray', label='Base Model')
for i, txt in enumerate(integral_percent):
    ax.annotate(f'{txt:.2f}%', (iterations[i], integral_percent[i]), textcoords="offset points", xytext=(0,10), ha='center', color='gray', fontsize=7)

# Data for "Afterburner 3B-SFT" - Integral%
integral_percent = np.array([6.93, 8.15, 8.15, 8.27, 8.44, 8.48, 8.83, 9.01, 9.01, 9.01, 9.05])
ax.plot(iterations, integral_percent, marker='o', linestyle='-', color='darkorange', label='Afterburner SFT')
for i, txt in enumerate(integral_percent):
    ax.annotate(f'{txt:.2f}%', (iterations[i], integral_percent[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=7)

# Data for "Afterburner 3B-DPO" - Integral%
integral_percent = np.array([8.49, 10.36, 10.57, 10.73, 11.03, 11.03, 11.11, 11.27, 11.27, 11.27, 11.27])
ax.plot(iterations, integral_percent, marker='o', linestyle='-', color='darkblue', label='Afterburner DPO')
for i, txt in enumerate(integral_percent):
    ax.annotate(f'{txt:.2f}%', (iterations[i], integral_percent[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=7)

# Data for "Afterburner 3B-GRPO" - Integral%
integral_percent = np.array([4.69, 9.04, 11.42, 12.36, 14.01, 14.85, 15.22, 16.01, 16.18, 16.18, 16.18])
ax.plot(iterations, integral_percent, marker='o', linestyle='-', color='forestgreen', label='Afterburner GRPO')
for i, txt in enumerate(integral_percent):
    ax.annotate(f'{txt:.2f}%', (iterations[i], integral_percent[i]), textcoords="offset points", xytext=(0,10), ha='center', color='forestgreen', fontsize=7)

ax.set_title('Time% v.s. Iterations', fontsize=14)
ax.set_ylabel('Time (%)', fontsize=12) # Added (%) for clarity
ax.set_xticks(iterations)
ax.tick_params(axis='both', which='major', labelsize=10)
# Individual legend removed
ax.grid(True, linestyle='--', alpha=0.5)


# Create a single legend for the entire figure
# Get handles and labels from one of the subplots (e.g., the first one)
handles, labels = axs[0].get_legend_handles_labels()
# Place the legend at the bottom center of the figure
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.02), fancybox=True, ncol=4, fontsize=15)

# Adjust layout to prevent overlapping titles/labels and to make space for the single legend
# The rect parameter is [left, bottom, right, top]
# Adjusted bottom (0.08) to make more space for the figure-level legend
plt.tight_layout(rect=[0, 0.08, 1, 1]) 
# fig.suptitle('Performance Metrics vs. Iterations for Different Models', fontsize=18, y=1.0) # Optional overall title, adjusted y

plt.savefig('apps_combine.pdf', dpi=1200, bbox_inches='tight') # Changed filename
plt.savefig('apps_combine.png', dpi=600, bbox_inches='tight') # Changed filename
plt.show()
