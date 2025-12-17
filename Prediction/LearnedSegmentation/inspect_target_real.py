import pickle
import numpy as np
import xarray as xr
from pathlib import Path

path = Path("/media/caron/X9 Pro/travaille/Thèse/csv/departement-06-alpes-maritimes/raster/2x2/departement-06-alpes-maritimesInfluence.pkl")

try:
    with open(path, 'rb') as f:
        data = pickle.load(f)
        
    print(f"Type: {type(data)}")
    if isinstance(data, np.ndarray):
        print(f"Shape: {data.shape}")
    elif isinstance(data, xr.DataArray) or isinstance(data, xr.Dataset):
        print(data)
        
except Exception as e:
    print(f"Error: {e}")
