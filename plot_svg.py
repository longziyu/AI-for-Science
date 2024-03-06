import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data from the uploaded Excel file
data_path = '/mnt/data/数据.xlsx'
data = pd.read_excel(data_path)

# Convert 'Training Time' and 'Time Varience' to numeric values
data['Training Time Numeric'] = data['Training Time'].apply(lambda x: int(x.replace('k', '')))
data['Time Varience Numeric'] = data['Time Varience'].apply(lambda x: int(x.replace('k', '')))

# Plotting setup
plt.figure(figsize=(10, 6))
colors = plt.cm.tab20(np.linspace(0, 1, data.shape[0]))

# Plotting ellipses
for i, row in data.iterrows():
    ellipse = plt.matplotlib.patches.Ellipse(xy=(row['Training Time Numeric'], row['SR']),
                                              width=row['Time Varience Numeric']*2,
                                              height=row['SR Varience']*2,
                                              edgecolor=colors[i],
                                              facecolor=colors[i],
                                              alpha=0.5)
    plt.gca().add_patch(ellipse)

# Axis adjustments
plt.xlim(120, 305)  # Adjusted based on user's request
plt.ylim(0, 0.45)  # Adjusted based on user's request
plt.xlabel('Training Time (in thousand timesteps)', fontsize=14)
plt.ylabel('SR (%)', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(np.arange(0, 0.5, 0.05), ['{:.0f}%'.format(y * 100) for y in np.arange(0, 0.5, 0.05)], fontsize=12)

plt.grid(True)

# Adding legend
handles = [plt.Line2D([0], [0], marker='o', color='w', label=row['Method'],
                      markerfacecolor=colors[i], markersize=10) for i, row in data.iterrows()]
plt.legend(handles=handles, title="Methods", loc='upper right', fontsize=12, title_fontsize=14)

plt.tight_layout()

# Saving the plot as an SVG file
svg_file_path = '/mnt/data/adjusted_plot.svg'
plt.savefig(svg_file_path, format='svg')
