import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 22-10 Role, 1-23 Evidence Object or 3-31, 12-24 Definiendum (Exploratory)

experiment_id = "Role"
model_id = 'mistralaiMistral-7B-v0.1'
layer_depth = 28
head_idx = 16

facet_mapping = {
    'Definiendum': {
        'name': 'facet-1',
        'tokens': ['<s>', '▁Section', '▁', '2', '(', 's', ')', '▁of', '▁Industrial', '▁Dis', 'put', 'es', '▁Act', ':', "▁'", 'work', 'man', "'", '▁means', '▁any', '▁person', '▁(', 'including', '▁an', '▁app', 'rent', 'ice', ')', '▁employed', '▁in', '▁any', '▁industry', '▁to', '▁do', '▁any', '▁manual', ',', '▁un', 'sk', 'illed', ',', '▁skilled', ',', '▁technical', ',', '▁operational', ',', '▁cler', 'ical', '▁or', '▁super', 'vis', 'ory', '▁work', '▁for', '▁hire', '▁or', '▁reward'],
        'facet_indices': [15, 16]
    },
    'Role': {
        'name': 'facet-2',
        'tokens': ['<s>', '▁Section', '▁', '7', '(', '1', ')', '▁of', '▁Pre', 'vention', '▁of', '▁Money', '-', 'L', 'au', 'nd', 'ering', '▁Act', ',', '▁', '2', '0', '0', '2', ':', '▁The', '▁Central', '▁Government', '▁shall', '▁provide', '▁each', '▁Ad', 'jud', 'icating', '▁Authority', '▁with', '▁such', '▁officers', '▁and', '▁employees', '▁as', '▁that', '▁Government', '▁may', '▁think', '▁fit'],
        'facet_indices': [31, 32, 33, 34, 37, 39]
    },
    'Permissible Action': {
        'name': 'facet-3',
        'tokens': ['<s>', '▁Section', '▁', '4', '(', 'a', ')', '▁of', '▁Sec', 'ur', 'it', 'isation', '▁Act', ':', '▁...', 'take', '▁possession', '▁of', '▁the', '▁secured', '▁assets', '▁of', '▁the', '▁borrow', 'er', '▁including', '▁the', '▁right', '▁to', '▁transfer', '▁by', '▁way', '▁of', '▁lease', ',', '▁assignment', '▁or', '▁sale', '▁for', '▁real', 'ising', '▁the', '▁secured', '▁asset'],
        'facet_indices': [15, 16, 17, 18, 19, 20, 21, 22, 23, 24] 
    },
    'Prohibitory Action': {
        'name': 'facet-4',
        'tokens': ['<s>', '▁Section', '▁', '3', '▁of', '▁Motor', '▁V', 'eh', 'icles', '▁Act', ':', '▁No', '▁person', '▁shall', '▁drive', '▁a', '▁motor', '▁vehicle', '▁in', '▁any', '▁public', '▁place', '▁unless', '▁he', '▁holds', '▁an', '▁effective', '▁driving', '▁lic', 'ence', '▁issued', '▁to', '▁him', '▁author', 'ising', '▁him', '▁to', '▁drive', '▁the', '▁vehicle'],
        'facet_indices': [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
    },
    'Condition': {
        'name': 'facet-5',
        'tokens': ['<s>', '▁Section', '▁', '2', '5', 'F', '▁of', '▁Industrial', '▁Dis', 'put', 'es', '▁Act', ':', '▁No', '▁work', 'man', '▁employed', '▁in', '▁any', '▁industry', '▁who', '▁has', '▁been', '▁in', '▁continuous', '▁service', '▁for', '▁not', '▁less', '▁than', '▁one', '▁year', '▁under', '▁an', '▁employer', '▁shall', '▁be', '▁ret', 'ren', 'ched', '▁by', '▁that', '▁employer', '▁until'],
        'facet_indices': [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]
    },
    'Evidence Object': {
        'name': 'facet-6',
        'tokens': ['<s>', '▁Section', '▁', '2', '5', 'N', '(', '7', ')', '▁of', '▁ID', 'A', ':', '▁Where', '▁no', '▁application', '▁for', '▁permission', '▁under', '▁sub', '-', '▁section', '▁(', '1', ')', '▁is', '▁made', ',', '▁or', '▁where', '▁the', '▁permission', '▁for', '▁any', '▁ret', 'rench', 'ment', '▁has', '▁been', '▁refused', ',', '▁such', '▁ret', 'rench', 'ment', '▁shall', '▁be', '▁deemed', '▁to', '▁be', '▁illegal', '▁from', '▁the', '▁date', '▁on', '▁which', '▁the', '▁notice', '▁of', '▁ret', 'rench', 'ment', '▁was', '▁given', '▁to', '▁the', '▁work', 'man', '▁and'],
        'facet_indices': [57, 58, 59, 60, 61]
    },
    'Theme Facts': {
        'name': 'thematic',
        'tokens': ['<s>', '▁At', '▁an', '▁und', 'et', 'erm', 'ined', '▁time', '▁between', '▁', '1', '8', ':', '0', '0', '▁on', '▁May', '▁', '1', '2', ',', '▁', '2', '0', '1', '7', '▁and', '▁', '0', '6', ':', '0', '0', '▁on', '▁May', '▁', '1', '3', ',', '▁', '2', '0', '1', '7', ',', '▁at', '▁the', '▁parked', '▁delivery', '▁vehicle', '▁brand', 'ed', '▁Pe', 'uge', 'ot', '▁Box', 'er', ',', '▁an', '▁unknown', '▁individual', '▁used', '▁an', '▁un', 'ident', 'ified', '▁object', '▁to', '▁p', 'ry', '▁open', '▁the', '▁locks', '▁of', '▁the', '▁driver', '’', 's', '▁door', ',', '▁the', '▁passenger', '▁door', ',', '▁and', '▁then', '▁the', '▁cargo', '▁space', '.', '▁The', '▁individual', '▁entered', '▁the', '▁vehicle', '▁and', '▁stole', '▁from', '▁it', '▁a', '▁car', '▁radio', ',', '▁a', '▁demol', 'ition', '▁hammer', ',', '▁an', '▁electric', '▁saw', ',', '▁a', '▁dr', 'ill', ',', '▁and', '▁other', '▁work', '▁tools', ',', '▁all', '▁valued', '▁at', '▁', '8', ',', '7', '0', '0', '▁C', 'Z', 'K', '▁[...]', '▁By', '▁dam', 'aging', '▁the', '▁door', '▁lock', ',', '▁he', '▁caused', '▁damage', '▁worth', '▁', '3', ',', '5', '0', '0', '▁C', 'Z', 'K', '.', '▁The', '▁stolen', '▁items', '▁were', '▁sold', '▁to', '▁unknown', '▁persons', '.'],
        'facet_indices': [92, 93, 94, 95, 96, 97, 98]
    }
}

filepath = f"{facet_mapping[experiment_id]['name']}_{model_id}_attentions_filtered.npy"
data = np.load(filepath)

# Create a DataFrame for better readability
df = pd.DataFrame(data[layer_depth][head_idx])

# Get the tokens for the 'Evidence Object' (facet-6)
tokens_stripped = [
    '<space>' if token == '▁' else token.lstrip('_▁')
    for token in facet_mapping[experiment_id]['tokens']
]

print(f"Length of 'tokens' = {len(tokens_stripped)}")
print(f"Shape of pd.DataFrame = {df.shape}")

# Ensure that the number of tokens matches the DataFrame's size
if len(tokens_stripped) != df.shape[0]:
    raise ValueError("The number of tokens must match the DataFrame dimensions.")

# Create a mask for the upper triangle (excluding the diagonal)
mask = np.triu(np.ones_like(df, dtype=bool), k=1)  # k=1 excludes the diagonal

# Define a function for annotating only values > 0.05 and with "+" prefix for positive floats
def custom_annot(val):
    if val > 0.0999:
        return f"{val:.2f}" if val > 0 else f"{val:.2f}"
    return ""

# Create an annotation DataFrame based on your custom annotation rule
annot_df = df.applymap(custom_annot)

# Plotting the heatmap with tokens as axes labels
plt.figure(figsize=(10, 6))
ax = sns.heatmap(df, annot=annot_df, mask=mask, cmap='viridis', fmt='', cbar=True)

# Explicitly set the ticks to match the number of tokens
ax.set_xticks(np.arange(df.shape[1]) + 0.5)  # Align ticks with the center of the cells
ax.set_yticks(np.arange(df.shape[0]) + 0.5)

# Set the x and y axis labels to the tokens
ax.set_xticklabels(tokens_stripped, rotation=0, fontsize=15)
ax.set_yticklabels(tokens_stripped, rotation=0, fontsize=15)

# Select indices to bold
bold_indices = facet_mapping[experiment_id]['facet_indices']

# Bold the specified xticks
for i, label in enumerate(ax.get_xticklabels()):
    if i in bold_indices:
        label.set_fontweight('bold')
        label.set_color('red')

# Bold the specified yticks
for i, label in enumerate(ax.get_yticklabels()):
    if i in bold_indices:
        label.set_fontweight('bold')
        label.set_color('red')

# Add plot title, x-axis, and y-axis labels
ax.set_xlabel('Keys', fontsize=15)  # x-axis title
ax.set_ylabel('Queries', fontsize=15)  # y-axis title

# Remove any alpha transparency
ax.xaxis.label.set_alpha(1.0)
ax.yaxis.label.set_alpha(1.0)
ax.title.set_alpha(1.0)

# Show the heatmap
plt.show()