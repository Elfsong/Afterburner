import matplotlib.pyplot as plt
import numpy as np

# Overall figure and subplots
# Create a figure with 1 row and 4 columns of subplots
# Adjust figsize to accommodate all subplots; you might need to tweak this
fig, axs = plt.subplots(2, 2, figsize=(14, 6)) # Increased width and height slightly for better annotation spacing

plt.style.use('seaborn-v0_8-whitegrid') # Apply style once for the whole figure

iterations = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) # X-axis values: Number of iterations

# --- Plot 1: Pass@1 v.s. Iterations ---
ax = axs[0, 0] # Select the first subplot

# Data for "Base Model" - Pass@1
percent_pass1_base = np.array([27.99, 28.33, 28.67, 28.67, 29.00, 29.08, 29.17, 29.24, 29.33, 29.33, 29.33])
ax.plot(iterations, percent_pass1_base, marker='o', linestyle='-', color='gray', label='Base Model')
for i, txt in enumerate(percent_pass1_base):
    ax.annotate(f'{txt:.2f}%', (iterations[i], percent_pass1_base[i]), textcoords="offset points", xytext=(0,6), ha='center', fontsize=7, color='gray')

# Data for "Afterburner 3B-SFT" - Pass@1
percent_pass1_sft = np.array([46.00, 46.00, 46.33, 47.00, 48.33, 48.33, 48.67, 48.67, 48.67, 48.67, 48.67])
ax.plot(iterations, percent_pass1_sft, marker='o', linestyle='-', color='darkorange', label='Afterburner SFT')
for i, txt in enumerate(percent_pass1_sft):
    ax.annotate(f'{txt:.2f}%', (iterations[i], percent_pass1_sft[i]), textcoords="offset points", xytext=(0,6), ha='center', fontsize=7)

# Data for "Afterburner 3B-DPO" - Pass@1
percent_pass1_dpo = np.array([43.00, 50.00, 51.33, 51.50, 51.67, 51.67, 51.67, 51.67, 51.67, 51.67, 51.67])
ax.plot(iterations, percent_pass1_dpo, marker='o', linestyle='-', color='darkblue', label='Afterburner DPO')
for i, txt in enumerate(percent_pass1_dpo):
    ax.annotate(f'{txt:.2f}%', (iterations[i], percent_pass1_dpo[i]), textcoords="offset points", xytext=(0,6), ha='center', fontsize=7)

# Data for "Afterburner 3B-GRPO" - Pass@1
percent_pass1_grpo = np.array([37.33, 48.33, 52.00, 54.50, 57.00, 58.17, 59.34, 60.50, 61.18, 61.67, 61.67])
ax.plot(iterations, percent_pass1_grpo, marker='o', linestyle='-', color='forestgreen', label='Afterburner GRPO')
for i, txt in enumerate(percent_pass1_grpo):
    ax.annotate(f'{txt:.2f}%', (iterations[i], percent_pass1_grpo[i]), textcoords="offset points", xytext=(0,6), ha='center', fontsize=7, color='forestgreen')

ax.set_title('Pass@1 v.s. Iterations', fontsize=14) # Adjusted fontsize
ax.set_ylabel('Pass@1 (%)', fontsize=12) # Added (%) for clarity, adjusted fontsize
ax.set_xticks(iterations)
ax.tick_params(axis='both', which='major', labelsize=10) # Adjusted fontsize
# Individual legend removed
ax.grid(True, linestyle='--', alpha=0.5)


# --- Plot 2: Time% v.s. Iterations ---
ax = axs[0, 1] # Select the second subplot

# Data for "Base Model" - Time%
percent_time_base = np.array([12.04, 12.74, 13.08, 13.25, 13.42, 13.50, 13.59, 13.68, 13.69, 13.69, 13.71])
ax.plot(iterations, percent_time_base, marker='o', linestyle='-', color='gray', label='Base Model')
for i, txt in enumerate(percent_time_base):
    ax.annotate(f'{txt:.2f}%', (iterations[i], percent_time_base[i]), textcoords="offset points", xytext=(0,6), ha='center', fontsize=7, color='gray')

# Data for "Afterburner 3B-SFT" - Time%
percent_time_sft = np.array([25.04, 26.00, 26.40, 26.55, 26.70, 26.72, 26.74, 26.76, 26.78, 26.78, 26.78])
ax.plot(iterations, percent_time_sft, marker='o', linestyle='-', color='darkorange', label='Afterburner SFT')
for i, txt in enumerate(percent_time_sft):
    ax.annotate(f'{txt:.2f}%', (iterations[i], percent_time_sft[i]), textcoords="offset points", xytext=(0,-15), ha='center', fontsize=7)

# Data for "Afterburner 3B-DPO" - Time%
percent_time_dpo = np.array([24.61, 26.43, 27.93, 28.03, 28.47, 29.04, 29.46, 29.90, 30.42, 30.90, 30.92])
ax.plot(iterations, percent_time_dpo, marker='o', linestyle='-', color='darkblue', label='Afterburner DPO')
for i, txt in enumerate(percent_time_dpo):
    ax.annotate(f'{txt:.2f}%', (iterations[i], percent_time_dpo[i]), textcoords="offset points", xytext=(0,6), ha='center', fontsize=7)

# Data for "Afterburner 3B-GRPO" - Time%
percent_time_grpo = np.array([31.22, 36.69, 38.41, 39.64, 40.83, 42.35, 43.49, 44.30, 44.82, 45.17, 45.17])
ax.plot(iterations, percent_time_grpo, marker='o', linestyle='-', color='forestgreen', label='Afterburner GRPO')
for i, txt in enumerate(percent_time_grpo):
    ax.annotate(f'{txt:.2f}%', (iterations[i], percent_time_grpo[i]), textcoords="offset points", xytext=(0,6), ha='center', fontsize=7, color='forestgreen')

ax.set_title('Beyond_T v.s. Iterations', fontsize=14)
ax.set_ylabel('Beyond_T (%)', fontsize=12) # Added (%) for clarity
ax.set_xticks(iterations)
ax.tick_params(axis='both', which='major', labelsize=10)
# Individual legend removed
ax.grid(True, linestyle='--', alpha=0.5)


# --- Plot 3: Memory% v.s. Iterations ---
ax = axs[1, 0] # Select the third subplot

# Data for "Base Model" - Memory%
percent_mem_base = np.array([13.24, 13.73, 14.22, 14.46, 14.71, 14.84, 14.97, 15.10, 15.19, 15.19, 15.19])
ax.plot(iterations, percent_mem_base, marker='o', linestyle='-', color='gray', label='Base Model')
for i, txt in enumerate(percent_mem_base):
    ax.annotate(f'{txt:.2f}%', (iterations[i], percent_mem_base[i]), textcoords="offset points", xytext=(0,6), ha='center', fontsize=7, color='gray')

# Data for "Afterburner 3B-SFT" - Memory%
percent_mem_sft = np.array([23.87, 23.80, 24.20, 24.55, 24.90, 25.00, 25.10, 25.19, 25.29, 25.30, 25.30])
ax.plot(iterations, percent_mem_sft, marker='o', linestyle='-', color='darkorange', label='Afterburner SFT')
for i, txt in enumerate(percent_mem_sft):
    ax.annotate(f'{txt:.2f}%', (iterations[i], percent_mem_sft[i]), textcoords="offset points", xytext=(0,-15), ha='center', fontsize=7)

# Data for "Afterburner 3B-DPO" - Memory%
percent_mem_dpo = np.array([20.53, 24.29, 26.31, 26.74, 27.17, 27.40, 27.67, 27.68, 27.68, 27.68, 27.69])
ax.plot(iterations, percent_mem_dpo, marker='o', linestyle='-', color='darkblue', label='Afterburner DPO')
for i, txt in enumerate(percent_mem_dpo):
    ax.annotate(f'{txt:.2f}%', (iterations[i], percent_mem_dpo[i]), textcoords="offset points", xytext=(0,6), ha='center', fontsize=7)

# Data for "Afterburner 3B-GRPO" - Memory%
percent_mem_grpo = np.array([25.14, 30.21, 34.17, 37.45, 40.72, 42.55, 44.39, 46.22, 47.65, 48.05, 48.05])
ax.plot(iterations, percent_mem_grpo, marker='o', linestyle='-', color='forestgreen', label='Afterburner GRPO')
for i, txt in enumerate(percent_mem_grpo):
    ax.annotate(f'{txt:.2f}%', (iterations[i], percent_mem_grpo[i]), textcoords="offset points", xytext=(0,6), ha='center', fontsize=7, color='forestgreen')

ax.set_title('Beyond_M v.s. Iterations', fontsize=14)
ax.set_ylabel('Beyond_M (%)', fontsize=12) # Added (%) for clarity
ax.set_xticks(iterations)
ax.tick_params(axis='both', which='major', labelsize=10)
# Individual legend removed
ax.grid(True, linestyle='--', alpha=0.5)


# --- Plot 4: Integral% v.s. Iterations ---
ax = axs[1, 1] # Select the fourth subplot

# Data for "Base Model" - Integral%
integral_percent_base = np.array([10.29, 10.77, 11.25, 11.49, 11.72, 11.83, 11.96, 12.07, 12.19, 12.19, 12.19])
ax.plot(iterations, integral_percent_base, marker='o', linestyle='-', color='gray', label='Base Model')
for i, txt in enumerate(integral_percent_base):
    ax.annotate(f'{txt:.2f}%', (iterations[i], integral_percent_base[i]), textcoords="offset points", xytext=(0,6), ha='center', fontsize=7, color='gray')

# Data for "Afterburner 3B-SFT" - Integral%
integral_percent_sft = np.array([21.01, 21.09, 21.50, 21.88, 22.25, 22.31, 22.38, 22.44, 22.50, 22.50, 22.50])
ax.plot(iterations, integral_percent_sft, marker='o', linestyle='-', color='darkorange', label='Afterburner SFT')
for i, txt in enumerate(integral_percent_sft):
    ax.annotate(f'{txt:.2f}%', (iterations[i], integral_percent_sft[i]), textcoords="offset points", xytext=(0,6), ha='center', fontsize=7)

# Data for "Afterburner 3B-DPO" - Integral%
integral_percent_dpo = np.array([19.13, 26.25, 27.05, 27.42, 27.95, 28.11, 28.51, 29.51, 29.51, 29.51, 29.51])
ax.plot(iterations, integral_percent_dpo, marker='o', linestyle='-', color='darkblue', label='Afterburner DPO')
for i, txt in enumerate(integral_percent_dpo):
    ax.annotate(f'{txt:.2f}%', (iterations[i], integral_percent_dpo[i]), textcoords="offset points", xytext=(0,6), ha='center', fontsize=7)

# Data for "Afterburner 3B-GRPO" - Integral%
integral_percent_grpo = np.array([16.24, 24.81, 29.44, 30.85, 33.56, 35.48, 37.09, 38.01, 38.62, 38.95, 38.96])
ax.plot(iterations, integral_percent_grpo, marker='o', linestyle='-', color='forestgreen', label='Afterburner GRPO')
for i, txt in enumerate(integral_percent_grpo):
    ax.annotate(f'{txt:.2f}%', (iterations[i], integral_percent_grpo[i]), textcoords="offset points", xytext=(0,6), ha='center', fontsize=7, color='forestgreen')

ax.set_title('Beyond_I v.s. Iterations', fontsize=14)
ax.set_ylabel('Beyond_I (%)', fontsize=12) # Added (%) for clarity
ax.set_xticks(iterations)
ax.tick_params(axis='both', which='major', labelsize=10)
# Individual legend removed
ax.grid(True, linestyle='--', alpha=0.5)


# Create a single legend for the entire figure
# Get handles and labels from one of the subplots (e.g., the first one)
handles, labels = axs[0, 0].get_legend_handles_labels()
# Place the legend at the bottom center of the figure
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.02), fancybox=True, ncol=4, fontsize=15)

# Adjust layout to prevent overlapping titles/labels and to make space for the single legend
# The rect parameter is [left, bottom, right, top]
# Adjusted bottom (0.08) to make more space for the figure-level legend
plt.tight_layout(rect=[0, 0.08, 1, 1]) 
# fig.suptitle('Performance Metrics vs. Iterations for Different Models', fontsize=18, y=1.0) # Optional overall title, adjusted y

plt.savefig('venus_combine.pdf', dpi=1200, bbox_inches='tight') # Changed filename
plt.savefig('venus_combine.png', dpi=600, bbox_inches='tight') # Changed filename
plt.show()
