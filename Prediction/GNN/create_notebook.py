import nbformat as nbf

nb = nbf.v4.new_notebook()

code_cells = [
    """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wasserstein_distance
from sklearn.preprocessing import StandardScaler

# Paths to the datasets
path_bdiff = '/home/caron/Bureau/ST-GNN-for-wildifre-prediction/Prediction/GNN/bdiff/firepoint/2x2/train/occurence_default/df_train_full_departement_0_None_node.pkl'
path_firemen = '/home/caron/Bureau/ST-GNN-for-wildifre-prediction/Prediction/GNN/firemen/firepoint/2x2/train/occurence_01_06_25/df_train_full_3_0_risk-size-watershed-degree-a3-r5-t0.3_node.pkl'

# Load datasets
df_bdiff = pd.read_pickle(path_bdiff)
df_firemen = pd.read_pickle(path_firemen)

print(f"BDIFF shape: {df_bdiff.shape}")
print(f"FIREMEN shape: {df_firemen.shape}")""",
    
    """# Targets to compare
targets_bdiff = ['nbsinister', 'burnedareaRoot']
targets_firemen = ['nbsinister', 'timeintervention', 'ressource']

# Filter values > 0 to focus on actual events
df_bdiff_filtered = df_bdiff[df_bdiff['nbsinister'] > 0].copy() if 'nbsinister' in df_bdiff.columns else df_bdiff
df_firemen_filtered = df_firemen[df_firemen['nbsinister'] > 0].copy() if 'nbsinister' in df_firemen.columns else df_firemen

print(f"BDIFF filtered shape (>0): {df_bdiff_filtered.shape}")
print(f"FIREMEN filtered shape (>0): {df_firemen_filtered.shape}")""",

    """# Standardize the data so we can compare the shape of the distributions properly
scaler = StandardScaler()

dict_bdiff_std = {}
for col in targets_bdiff:
    vals = df_bdiff_filtered[col].dropna().values.reshape(-1, 1)
    if len(vals) > 0:
        dict_bdiff_std[col] = scaler.fit_transform(vals).flatten()

dict_firemen_std = {}
for col in targets_firemen:
    vals = df_firemen_filtered[col].dropna().values.reshape(-1, 1)
    if len(vals) > 0:
        dict_firemen_std[col] = scaler.fit_transform(vals).flatten()""",

    """# Plot the distributions
fig, axes = plt.subplots(len(targets_bdiff), len(targets_firemen), figsize=(18, 10))

results = []

for i, tb in enumerate(targets_bdiff):
    for j, tf in enumerate(targets_firemen):
        ax = axes[i, j]
        
        data_b = dict_bdiff_std[tb]
        data_f = dict_firemen_std[tf]
        
        # Calculate Wasserstein distance
        dist = wasserstein_distance(data_b, data_f)
        results.append({'bdiff_target': tb, 'firemen_target': tf, 'wasserstein_distance': dist})
        
        sns.kdeplot(data_b, ax=ax, label=f'BDIFF: {tb}', fill=True, alpha=0.5)
        sns.kdeplot(data_f, ax=ax, label=f'FIREMEN: {tf}', fill=True, alpha=0.5)
        
        ax.set_title(f"{tb} vs {tf}\\nDist: {dist:.4f}")
        ax.legend()

plt.tight_layout()
plt.show()""",

    """# Summarize results
results_df = pd.DataFrame(results)

# For each BDIFF target, find the closest FIREMEN target
best_matches = results_df.loc[results_df.groupby('bdiff_target')['wasserstein_distance'].idxmin()]

print("=== All Distances ===")
print(results_df.sort_values(['bdiff_target', 'wasserstein_distance']))
print("\\n=== Best Matches ===")
print(best_matches)"""
]

nb['cells'] = [nbf.v4.new_markdown_cell("# Comparaison des distributions entre BDIFF et FIREMEN\nL'objectif est de trouver quelle variable de la base firemen (nbsinister, timeintervention, ressource) a la distribution la plus proche des variables de la base bdiff (nbsinister, burnedareaRoot). Les données sont standardisées pour comparer la forme des distributions.")] + [nbf.v4.new_code_cell(c) for c in code_cells]

with open('/home/caron/Bureau/ST-GNN-for-wildifre-prediction/Prediction/GNN/compare_distribution_bdiff_firemen.ipynb', 'w') as f:
    nbf.write(nb, f)
