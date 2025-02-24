# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt

# Load the Excel file
file_path = "all_results_NAF.xlsx"  # Replace with your actual file path
df = pd.read_excel(file_path)

df = df.sort_values(by="Mean_Match", ascending=True)

# Plot the data
plt.figure(figsize=(10, 6))
plt.errorbar(df["Mean_Match"], df["Prediction_File"], xerr=df["Std_Dev"], fmt='o', 
             color='darkblue', ecolor='red', capsize=5, markersize=8, label="Mean Match")

# Labels and title
plt.xlabel("Mean Match (%)")
plt.ylabel("Model Name")
plt.title("Mean PoS Tagging Accuracy across phrases - NAF6195")
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.legend()

# Show the plot
plt.tight_layout()
plt.savefig('plot_RCPTPextended_NAF.png', dpi=300)
plt.show()
