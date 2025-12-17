
import pickle
import sys
from pathlib import Path

path = "/media/caron/X9 Pro/travaille/Thèse/csv/departement-06-alpes-maritimes/raster/2x2/datacube.pkl"

try:
    with open(path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Type: {type(data)}")
    if isinstance(data, dict):
        print(f"Keys: {list(data.keys())}")
        for k, v in data.items():
            print(f"Key: {k}, Type: {type(v)}")
            if hasattr(v, 'shape'):
                print(f"Shape: {v.shape}")
    else:
        print(data)

except Exception as e:
    print(f"Error: {e}")
