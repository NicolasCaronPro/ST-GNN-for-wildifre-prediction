
import sys
import os
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from unittest.mock import MagicMock, patch
import matplotlib.pyplot as plt

# Add the parent directory to sys.path to import GNN modules
sys.path.append('/home/caron/Bureau/ST-GNN-for-wildifre-prediction/Prediction')

# Mocking dependencies that are hard to setup
dataloader_mock = MagicMock()
sys.modules['GNN.dataloader'] = dataloader_mock
# Ensure read_object is available in the mock
dataloader_mock.read_object = MagicMock()
dataloader_mock.save_object = MagicMock()
dataloader_mock.check_and_create_path = MagicMock()
dataloader_mock.root_target = Path('/tmp')
dataloader_mock.rootDisk = Path('/tmp')

sys.modules['GNN.pytorch_model'] = MagicMock()
sys.modules['skimage'] = MagicMock()
sys.modules['skimage.io'] = MagicMock()
sys.modules['skimage.color'] = MagicMock()
sys.modules['skimage.filters'] = MagicMock()
sys.modules['skimage.measure'] = MagicMock()
sys.modules['skimage.morphology'] = MagicMock()
sys.modules['skimage.segmentation'] = MagicMock()
sys.modules['skimage.feature'] = MagicMock()
sys.modules['skimage.util'] = MagicMock()
# sys.modules['scipy'] = MagicMock() # Removed to avoid conflict with sklearn
sys.modules['scipy.ndimage'] = MagicMock()
sys.modules['scipy.spatial.distance'] = MagicMock()
sys.modules['tslearn.clustering'] = MagicMock()
sys.modules['GNN.tools'] = MagicMock()

# Import GraphStructure
from GNN.graph_structure import GraphStructure, iou_binary, to_binary_mask
import GNN.graph_structure as graph_structure

# Mock logger
graph_structure.logger = MagicMock()

# Mock read_object and save_object in graph_structure
graph_structure.read_object = MagicMock()
graph_structure.save_object = MagicMock()
graph_structure.check_and_create_path = MagicMock()
graph_structure.root_target = Path('/tmp')
graph_structure.rootDisk = Path('/tmp')

# Mocking specific methods used in create_geometry_with_watershed
GraphStructure._process_base_data = MagicMock()
GraphStructure._save_feature_image = MagicMock()
GraphStructure.my_watershed = MagicMock()
GraphStructure.create_cluster = MagicMock()
GraphStructure._post_process_result = MagicMock()

def test_plotting():
    print("Testing merge_adjacent_clusters replacement...")

    # Setup GraphStructure with search parameters
    gs = GraphStructure(
        scale=0,
        geo=None,
        maxDist=10,
        numNei=5,
        resolution='10m',
        graph_construct='watershed-size',
        sinister='fire',
        sinister_encoding='utf-8',
        dataset_name='dataset',
        train_departements=['06'],
        attempt='search',
        reduce=200, # Fixed reduce
        tol=0.3
    )
    
    # Mock data
    dept = '06'
    vec_base = ['watershed', 'size']
    path = Path('/tmp/test_plot')
    sinister = 'fire'
    dataset_name = 'dataset'
    sinister_encoding = 'utf-8'
    resolution = '10m'
    mask = np.ones((100, 100), dtype=bool)
    node_already_predicted = 0
    train_date = '2023-01-01'

    # Mock raster
    graph_structure.read_object.return_value = [np.zeros((100, 100))] # raster

    # Mock _process_base_data return (data, GT)
    data = np.random.rand(100, 100)
    GT = np.random.randint(0, 2, (100, 100))
    gs._process_base_data.return_value = (data, GT)

    # Mock my_watershed return
    gs.my_watershed.return_value = np.zeros((100, 100))

    # Mock create_cluster return (fr, pred)
    gs.create_cluster.return_value = (0.5, np.zeros((100, 100)))
    
    # Mock merge_adjacent_clusters
    graph_structure.merge_adjacent_clusters = MagicMock(return_value=np.zeros((100, 100)))
    
    # Mock check_and_create_path to just print
    graph_structure.check_and_create_path.side_effect = lambda p: print(f"Creating path: {p}")

    # Run the function
    try:
        gs.create_geometry_with_watershed(
            dept, vec_base, path, sinister, dataset_name,
            sinister_encoding, resolution, mask, node_already_predicted, train_date
        )
        print("create_geometry_with_watershed executed successfully.")
    except Exception as e:
        print(f"create_geometry_with_watershed failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Verify merge_adjacent_clusters call
    print(f"merge_adjacent_clusters call count: {graph_structure.merge_adjacent_clusters.call_count}")
    
if __name__ == "__main__":
    # Mock plt in graph_structure
    with patch('GNN.graph_structure.plt') as mock_plt:
        test_plotting()
        print(f"plt.savefig call count: {mock_plt.savefig.call_count}")
        if mock_plt.savefig.call_count > 0:
            print("Plot saving was attempted.")
        else:
            print("Plot saving was NOT attempted.")
