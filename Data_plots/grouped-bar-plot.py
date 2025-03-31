# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

# Read data from Excel files
df1 = pd.read_excel("../data/REF_Albuc_1.xlsx") #reference file
df2 = pd.read_excel("../data/NAF_reference.xlsx") #reference file

# Extract the required columns
values_list1 = df1["POS"].tolist()
list1 = df1["Lemma"].tolist()
values_list2 = df2["POS"].tolist()
list2 = df2["Lemma"].tolist()

# Combine the data into a single DataFrame for easier manipulation
data = {
    'POS': values_list1 + values_list2,
    'Lemma': list1 + list2,
    'Source': ['Albucasis'] * len(values_list1) + ['Vida de Sant Honorat'] * len(values_list2)
}
df = pd.DataFrame(data)

# Group the data by POS and Source
grouped = df.groupby(['POS', 'Source']).size().unstack(fill_value=0)

# Calculate the total number of tags for each source
total_tags_albucasis = grouped['Albucasis'].sum()
total_tags_vida_sant_honorat = grouped['Vida de Sant Honorat'].sum()

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Define colors for each group
colors = {'Albucasis': 'b', 'Vida de Sant Honorat': 'r'}

# Plot the bars
bar_width = 0.35
index = np.arange(len(grouped.index))
for i, (name, color) in enumerate(colors.items()):
    ax.bar(index + i * bar_width, grouped[name], bar_width, label=f'{name} ({total_tags_albucasis if name == "Albucasis" else total_tags_vida_sant_honorat})', color=color)

# Add some text for labels, title, and custom x-axis tick labels, etc.
ax.set_xlabel('Part of Speech Tags')
ax.set_ylabel('Count')
ax.set_title('Part-of-Speech Tagging distribution of the Texts: Albucasis and Vida de Sant Honorat')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(grouped.index, rotation=45)  # Rotate labels by 45 degrees

# Create custom legend handles
custom_handles = [
    mpatches.Patch(color='b', label=f'Albucasis (Total number: {total_tags_albucasis})'),
    mpatches.Patch(color='r', label=f'Vida de Sant Honorat (Total number: {total_tags_vida_sant_honorat})')
]

# Add the custom legend handles
ax.legend(handles=custom_handles)

plt.show()