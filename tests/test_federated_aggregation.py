import unittest
import torch
from Prediction.GNN.pytorch_model import FederatedLearningModel
from copy import deepcopy

class DummyModel:
    def __init__(self):
        self.received = None
    def update_weight(self, weight):
        self.received = weight

class AggregationTest(unittest.TestCase):
    def test_weighted_aggregation_sample_counts(self):
        base = DummyModel()
        model = FederatedLearningModel(base, features=[], aggregation_method='weighted')
        # prepare local weights
        w1 = {'param': torch.tensor([1.0, 2.0])}
        w2 = {'param': torch.tensor([3.0, 5.0])}
        sample_counts = [2, 6]
        model.aggregate_models([w1, w2], sample_counts)
        expected = (w1['param']*0.25 + w2['param']*0.75)
        self.assertTrue(torch.allclose(model.global_model.received['param'], expected))

if __name__ == '__main__':
    unittest.main()
