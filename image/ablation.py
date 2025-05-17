import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Data extracted from the image
data = {
    'Category': [
        'Afterburner 3B-SFT', 'Afterburner 3B-SFT', 'Afterburner 3B-SFT',
        'Afterburner 3B-DPO', 'Afterburner 3B-DPO', 'Afterburner 3B-DPO',
        'Afterburner 3B-GRPO', 'Afterburner 3B-GRPO', 'Afterburner 3B-GRPO'
    ],
    'Variation': [
        'Base', '- Remove feedback', '- Remove original code.',
        'Base', '- Remove feedback', '- Remove original code.',
        'Base', '- Remove feedback', '- Remove original code.'
    ],
    'Pass@1': [48.33, 46.33, 45.33, 51.67, 50.33, 47.33, 57.00, 52.51, 54.17],
    'Time%': [26.61, 25.41, 25.64, 28.45, 27.33, 25.32, 40.81, 34.15, 32.17],
    'Memory%': [24.39, 24.70, 26.17, 28.03, 26.73, 24.17, 40.68, 34.49, 33.25],
    'Integral%': [22.25, 21.43, 20.08, 27.89, 25.68, 22.01, 33.51, 29.87, 24.24]
}
df = pd.DataFrame(data)

# Unique categories and variations for plotting
categories = df['Category'].unique()
variations = df['Variation'].unique()

# Number of variations
n_variations = len(variations)
# Number of categories
n_categories = len(categories)

# Metrics to plot
metrics = ['Pass@1', 'Time%', 'Memory%', 'Integral%']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] # Colors for variations

for metric in metrics:
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Width of a bar group for each category
    bar_width_group = 0.6
    # Width of an individual bar
    bar_width_single = bar_width_group / n_variations
    
    # Positions for the groups of bars
    index = np.arange(n_categories)

    for i, variation in enumerate(variations):
        # Get current variation data for the metric
        metric_values = df[df['Variation'] == variation][metric].values
        # Calculate bar positions for the current variation
        # Offset each variation's bars within the group
        bar_positions = index - (bar_width_group / 2) + (i * bar_width_single) + (bar_width_single / 2)
        
        rects = ax.bar(bar_positions, metric_values, bar_width_single, label=variation, color=colors[i])
        
        # Add data labels on top of bars
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    ax.set_ylabel(metric)
    ax.set_title(f'{metric} by Model Category and Variation')
    ax.set_xticks(index)
    ax.set_xticklabels(categories, rotation=15, ha="right") # Rotate labels for better readability
    ax.legend(title='Variation')
    
    # Add some spacing around the plot
    plt.tight_layout(pad=2.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(f'ablation_{metric}.png', dpi=300)