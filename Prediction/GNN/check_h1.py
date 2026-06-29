import pandas as pd
from pathlib import Path
import pickle

p = Path("/home/caron/Bureau/ST-GNN-for-wildifre-prediction/Prediction/GNN/baseline/NetMLP_search_full_H1_Ope_2024_08_08/H1/test_stats_class.pkl")
if p.exists():
    with open(p, "rb") as f:
        df = pickle.load(f)
    print("Columns in H1:", [c for c in df.columns if 'prediction' in c])
