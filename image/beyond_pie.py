import matplotlib.pyplot as plt

# --- Data Dictionary (ensure this is correctly placed in your script) ---
data = {
    'Open Source Models': {
        'qwen_2_5_3b_instruct': {
            'Time': {'Beyond': 0.67, 'Medicore': 27.00, 'Below': 0.33, 'Failed': 72.00},
            'Memory': {'Beyond': 0.33, 'Medicore': 27.33, 'Below': 0.33, 'Failed': 72.00},
            'Integral': {'Beyond': 0.67, 'Medicore': 26.67, 'Below': 0.67, 'Failed': 72.00}
        },
        'qwen_2_5_coder_7b': {
            'Time': {'Beyond': 1.33, 'Medicore': 50.67, 'Below': 0.33, 'Failed': 47.67},
            'Memory': {'Beyond': 0.67, 'Medicore': 50.67, 'Below': 1.00, 'Failed': 47.67},
            'Integral': {'Beyond': 1.33, 'Medicore': 50.67, 'Below': 0.33, 'Failed': 47.67}
        },
        'qwen_2_5_7b_instruct': {
            'Time': {'Beyond': 1.67, 'Medicore': 58.33, 'Below': 0.67, 'Failed': 39.33},
            'Memory': {'Beyond': 1.00, 'Medicore': 58.33, 'Below': 1.33, 'Failed': 39.33},
            'Integral': {'Beyond': 1.33, 'Medicore': 58.00, 'Below': 1.67, 'Failed': 39.33}
        },
        'llama_4_scout_17b_16e_instruct': {
            'Time': {'Beyond': 3.00, 'Medicore': 59.33, 'Below': 0.33, 'Failed': 37.33},
            'Memory': {'Beyond': 2.00, 'Medicore': 60.67, 'Below': 0.33, 'Failed': 37.33},
            'Integral': {'Beyond': 1.67, 'Medicore': 60.67, 'Below': 0.67, 'Failed': 37.33}
        },
        'qwq_32b': {
            'Time': {'Beyond': 6.67, 'Medicore': 76.00, 'Below': 0.33, 'Failed': 17.00},
            'Memory': {'Beyond': 2.33, 'Medicore': 79.67, 'Below': 1.00, 'Failed': 17.00},
            'Integral': {'Beyond': 3.33, 'Medicore': 79.00, 'Below': 1.00, 'Failed': 17.00}
        }
    },
    'Closed Sourced Models': {
        'gpt_4o': {
            'Time': {'Beyond': 2.33, 'Medicore': 79.00, 'Below': 1.00, 'Failed': 17.67},
            'Memory': {'Beyond': 1.33, 'Medicore': 79.00, 'Below': 1.67, 'Failed': 17.67},
            'Integral': {'Beyond': 1.33, 'Medicore': 79.67, 'Below': 1.33, 'Failed': 17.67}
        },
        'claude_3_5_haiku': {
            'Time': {'Beyond': 4.67, 'Medicore': 61.67, 'Below': 0.33, 'Failed': 33.67},
            'Memory': {'Beyond': 2.00, 'Medicore': 64.00, 'Below': 0.33, 'Failed': 33.67},
            'Integral': {'Beyond': 2.67, 'Medicore': 63.33, 'Below': 0.67, 'Failed': 33.67}
        },
        'claude_3_7_sonnet': {
            'Time': {'Beyond': 5.67, 'Medicore': 80.67, 'Below': 0.33, 'Failed': 13.33},
            'Memory': {'Beyond': 2.67, 'Medicore': 83.33, 'Below': 0.33, 'Failed': 13.33},
            'Integral': {'Beyond': 3.33, 'Medicore': 82.00, 'Below': 1.00, 'Failed': 13.33}
        },
        'deepseek_v3': {
            'Time': {'Beyond': 5.33, 'Medicore': 80.67, 'Below': 0.67, 'Failed': 13.67},
            'Memory': {'Beyond': 3.33, 'Medicore': 82.67, 'Below': 0.33, 'Failed': 13.67},
            'Integral': {'Beyond': 3.00, 'Medicore': 81.67, 'Below': 1.67, 'Failed': 13.67}
        },
        'o4-mini': {
            'Time': {'Beyond': 7.00, 'Medicore': 82.00, 'Below': 0.00, 'Failed': 11.00},
            'Memory': {'Beyond': 3.33, 'Medicore': 85.33, 'Below': 0.67, 'Failed': 11.00},
            'Integral': {'Beyond': 4.00, 'Medicore': 84.33, 'Below': 0.67, 'Failed': 11.00}
        }
    },
    'Afterburner': {
        'afterburner (iteration 0)': {
            'Time': {'Beyond': 1.67, 'Medicore': 41.00, 'Below': 4.67, 'Failed': 52.67},
            'Memory': {'Beyond': 2.33, 'Medicore': 36.67, 'Below': 8.67, 'Failed': 52.67},
            'Integral': {'Beyond': 1.33, 'Medicore': 41.67, 'Below': 4.33, 'Failed': 52.67}
        },
        'afterburner (iteration 1)': {
            'Time': {'Beyond': 3.67, 'Medicore': 41.00, 'Below': 5.67, 'Failed': 49.67},
            'Memory': {'Beyond': 4.33, 'Medicore': 37.33, 'Below': 8.67, 'Failed': 49.67},
            'Integral': {'Beyond': 2.67, 'Medicore': 41.00, 'Below': 6.67, 'Failed': 49.67}
        },
        'afterburner (iteration 8)': {
            'Time': {'Beyond': 8.00, 'Medicore': 46.33, 'Below': 7.33, 'Failed': 38.33},
            'Memory': {'Beyond': 7.00, 'Medicore': 44.33, 'Below': 10.33, 'Failed': 38.33},
            'Integral': {'Beyond': 5.33, 'Medicore': 46.00, 'Below': 10.00, 'Failed': 38.33}
        }
    }
}
# -------------------------------------

model_name = 'gpt_4o' # Choose the model to plot
model_category = 'Closed Sourced Models' # Specify the category of the chosen model

# --- Robustly get model data ---
try:
    # Check if the category exists first
    if model_category not in data:
        print(f"Error: Model category '{model_category}' not found in data.")
        # Handle error appropriately, e.g., exit() or raise KeyError
        exit()
    # Then check if the model exists within the category
    if model_name not in data[model_category]:
        print(f"Error: Model '{model_name}' not found in category '{model_category}'.")
        # Handle error appropriately
        exit()
    model_data = data[model_category][model_name]
except Exception as e:
    print(f"An unexpected error occurred while accessing data: {e}")
    exit()
# --------------------------------

# Define metrics and categories
metrics = ['Time', 'Memory', 'Integral']
categories = ['Beyond', 'Medicore', 'Below', 'Failed']

# Define consistent colors for categories
category_colors = {
    'Beyond': '#4CAF50',    # Green
    'Medicore': '#FFC107',  # Amber
    'Below': '#FF9800',     # Orange
    'Failed': '#F44336'     # Red
}
# Ensure the colors list matches the order of categories
plot_colors = [category_colors.get(cat, '#CCCCCC') for cat in categories] # Use grey for unexpected categories

# Create a figure with 3 subplots (one for each metric)
fig, axes = plt.subplots(1, 3, figsize=(18, 6)) # Adjusted figsize for potentially better legend spacing
fig.suptitle(f'Performance Distribution for {model_name}', fontsize=16, y=0.98) # Adjusted title position slightly

plot_wedges = [] # To store wedges from the first plot for the legend

for i, metric in enumerate(metrics):
    # Check if metric data exists for the model
    if metric not in model_data:
        print(f"Warning: Metric '{metric}' not found for model '{model_name}'. Skipping.")
        axes[i].set_title(f'{metric}\n(No Data)')
        axes[i].set_xticks([]) # Remove ticks and labels for empty plots
        axes[i].set_yticks([])
        axes[i].spines['top'].set_visible(False) # Hide box outline
        axes[i].spines['right'].set_visible(False)
        axes[i].spines['bottom'].set_visible(False)
        axes[i].spines['left'].set_visible(False)
        continue # Skip to the next metric

    metric_data = model_data[metric]
    # Get values, ensuring all categories are present, defaulting to 0 if missing
    values = [metric_data.get(cat, 0) for cat in categories]

    # Create pie chart (donut style)
    # Explode the 'Failed' slice slightly if desired (optional)
    # explode = tuple([0.05 if cat == 'Failed' else 0 for cat in categories])

    wedges, texts, autotexts = axes[i].pie(
        values,
        autopct=lambda p: '{:.1f}%'.format(p) if p > 0 else '', # Show % only if > 0
        startangle=90,
        colors=plot_colors,
        pctdistance=0.85, # Place percentages inside wedges
        # explode=explode, # Uncomment to enable explode
        wedgeprops=dict(width=0.4, edgecolor='w') # Donut chart style
    )

    # Style the percentage text for better visibility
    plt.setp(autotexts, size=9, weight="bold", color="white")

    axes[i].set_title(metric, fontsize=12)

    # Store wedges from the first plot (they are needed for the figure legend)
    if i == 0:
        plot_wedges = wedges


# Add a single, shared legend for the whole figure
# Place it to the right of the subplots
if plot_wedges: # Only add legend if at least one plot was successful
    fig.legend(handles=plot_wedges, labels=categories, # Use categories for labels
               title="Categories",
               loc='center left', # Anchor point of the legend
               bbox_to_anchor=(0.91, 0.5), # Position: 0.91=slightly right of plots, 0.5=vertically centered
               fontsize=10,
               title_fontsize=11)

# Adjust layout to prevent overlap and make space for the legend
# This might need tuning depending on your display/output format
plt.subplots_adjust(left=0.05, right=0.90, top=0.9, bottom=0.1, wspace=0.2)

# Show the plot
plt.savefig(f'{model_name}_performance_distribution.png', dpi=300)