import pickle
import numpy as np
import xarray as xr
from pathlib import Path

path = Path("/media/caron/X9 Pro/travaille/Thèse/csv/departement-06-alpes-maritimes/raster/2x2/datacube.pkl")

try:
    with open(path, 'rb') as f:
        data = pickle.load(f)
        
    print(f"Type: {type(data)}")
    if isinstance(data, dict):
        print(f"Keys: {list(data.keys())[:10]}...")
        for k, v in data.items():
            print(f"Key: {k}, Type: {type(v)}")
            if hasattr(v, 'shape'):
                print(f"  Shape: {v.shape}")
            if hasattr(v, 'coords'):
                print(f"  Coords: {v.coords}")
            break # Inspect first item
            
    elif isinstance(data, xr.Dataset):
        print(data)
        
except Exception as e:
    print(f"Error: {e}")
