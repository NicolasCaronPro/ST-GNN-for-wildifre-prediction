import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import os
os.environ["MPLBACKEND"] = "Agg"

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path(__file__).resolve().parent.parent / 'GNN'))

from LearnedSegmentation.frontier_utils import ids_to_boundary_mask, connected_components_8

class TestFrontierWorkflow(unittest.TestCase):

    def test_ids_to_boundary_mask(self):
        L = np.array([
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [3, 3, 4, 4],
            [3, 3, 4, 4]
        ])
        # Expected boundary (4-connectivity)
        # (0,1) and (0,2) are diff -> (0,1) is 1, (0,2) is 1? No, diff_r at 1 is 1.
        # Let's trace manually.
        # diff_r:
        # 0 1 0
        # 0 1 0
        # 0 1 0
        # 0 1 0
        # diff_d:
        # 0 0 0 0
        # 1 1 1 1
        # 0 0 0 0
        
        # B |= diff_r (shifted)
        # B |= diff_d (shifted)
        
        B = ids_to_boundary_mask(L, connectivity=4)
        
        # Check specific points
        self.assertEqual(B[0, 1], 1) # Boundary between 1 and 2
        self.assertEqual(B[0, 2], 1) # Boundary between 1 and 2
        self.assertEqual(B[1, 1], 1)
        self.assertEqual(B[1, 2], 1)
        self.assertEqual(B[2, 1], 1)
        self.assertEqual(B[2, 2], 1)
        
        self.assertEqual(B[1, 0], 1) # Boundary between 1 and 3
        self.assertEqual(B[2, 0], 1)
        
        self.assertEqual(B[0, 0], 0) # Interior
        self.assertEqual(B[3, 3], 0) # Interior

    def test_connected_components_8(self):
        binary = np.array([
            [1, 1, 0, 1],
            [1, 1, 0, 1],
            [0, 0, 0, 0],
            [1, 1, 1, 1]
        ])
        # 1s are regions to label.
        # Component 1: top-left 2x2
        # Component 2: top-right 2x1
        # Component 3: bottom row
        
        labels = connected_components_8(binary)
        
        self.assertEqual(labels[0, 0], labels[0, 1])
        self.assertEqual(labels[0, 0], labels[1, 0])
        self.assertEqual(labels[0, 0], labels[1, 1])
        
        self.assertNotEqual(labels[0, 0], labels[0, 3])
        self.assertNotEqual(labels[0, 0], labels[3, 0])
        
        self.assertEqual(labels[0, 3], labels[1, 3])
        
        self.assertEqual(labels[3, 0], labels[3, 1])
        self.assertEqual(labels[3, 1], labels[3, 2])
        self.assertEqual(labels[3, 2], labels[3, 3])

    @patch('LearnedSegmentation.test.Tester.visualize_prediction')
    @patch('GNN.graph_structure.GraphStructure.find_closest_cluster_with_time_series')
    @patch('GNN.graph_structure.GraphStructure.find_closest_cluster')
    @patch('pickle.load')
    @patch('builtins.open')
    @patch('pathlib.Path.exists')
    def test_process_binary_frontier(self, mock_exists, mock_open, mock_pickle_load, mock_find_closest, mock_find_closest_ts, mock_vis):
        # Mock GNN.tools and GNN.arborescence to avoid import errors
        mock_gnn_tools = MagicMock()
        mock_gnn_tools.read_object = MagicMock(side_effect=[
            [np.zeros((10, 10))], # pred_clusters
            [{'time_series_clustering': np.zeros((10, 10))}] # datacube_target
        ])
        sys.modules['GNN.tools'] = mock_gnn_tools
        
        mock_gnn_arbo = MagicMock()
        mock_gnn_arbo.root_target = Path('/tmp')
        mock_gnn_arbo.rootDisk = Path('/tmp')
        sys.modules['GNN.arborescence'] = mock_gnn_arbo
        
        from LearnedSegmentation.test import Tester
        
        # Mock config
        mock_config = MagicMock()
        mock_config.get_scale.return_value = 3
        mock_config.get_graph_construct.return_value = "scale_3_tol_0.3_attempt_3_reduce_4"
        mock_config.get_n_clusters_node.return_value = 10
        mock_config.get_dataset_name.return_value = "dataset"
        
        # Mock Tester
        tester = Tester(mock_config, "model_path", {'type': 'UNet', 'params': {'task_type': 'binary'}})
        tester.score = pd.DataFrame()
        
        # Mock predictions (boundary mask)
        # 1 for boundary, 0 for interior
        predictions = np.zeros((1, 10, 10))
        predictions[0, 5, :] = 1 # Horizontal line
        
        y = np.zeros((1, 1, 10, 10))
        
        # Mock Path exists
        mock_exists.return_value = True
        
        # Mock pickle load (gs, encoder, ordinal_encoder)
        mock_gs = MagicMock()
        mock_gs.scale = 3
        mock_gs.base = "base"
        mock_gs.graph_method = "method"
        
        mock_encoder = MagicMock()
        mock_ordinal_encoder = MagicMock()
        
        mock_pickle_load.side_effect = [mock_gs, {}, [], mock_encoder, mock_ordinal_encoder] # gs, config, test_depts, encoder, ordinal_encoder
        
        # Mock encode_map
        tester.encode_map = MagicMock(return_value=np.zeros((10, 10)))
        
        # Run
        metrics = tester.process_binary_frontier(predictions, y, "dept_name")
        
        # Assertions
        self.assertTrue('Clustering_IoU' in metrics)
        mock_find_closest.assert_called() # Should be called if dept not in test_departements (mocked empty)

if __name__ == '__main__':
    unittest.main()
