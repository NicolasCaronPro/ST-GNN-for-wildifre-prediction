import pandas as pd
from pathlib import Path

path = Path("bdiff/firepoint/2x2/train/occurence_default_2024")
files = list(path.glob("df_train_full_*.pkl"))
if files:
    df = pd.read_pickle(files[0])
    print(f"File: {files[0]}")
    print(f"Unique departements: {df['departement'].nunique()}")
    print(f"Unique graph_ids: {df['graph_id'].nunique()}")
    print(f"Max graph_id: {df['graph_id'].max()}")
    print(f"Unique graph_ids list length: {len(df['graph_id'].unique())}")
else:
    print("Files not found locally.")
