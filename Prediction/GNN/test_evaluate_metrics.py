import sys
import os
import numpy as np
import pandas as pd

# Add the path to the GNN directory
sys.path.append('/home/caron/Bureau/ST-GNN-for-wildifre-prediction/Prediction/GNN')

from forecasting_models.sklearn.score import evaluate_metrics

def test_evaluate_metrics():
    print("Testing evaluate_metrics with new signature...")
    
    # Dummy data
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 1, 0, 2, 2])
    dates = np.array(['2023-01-01', '2023-01-01', '2023-01-01', '2023-01-02', '2023-01-02', '2023-01-02'])
    zones = np.array(['A', 'B', 'C', 'A', 'B', 'C'])
    
    # Test 1: Basic metrics (no dates/zones)
    print("\nTest 1: Basic metrics (no dates/zones)")
    results = evaluate_metrics(y_true, y_pred)
    print(f"IoU: {results['iou']}")
    print(f"F1: {results['f1']}")
    assert 'iou' in results
    assert np.isnan(results['score'])
    
    # Test 2: With dates and zones (monotonic score)
    print("\nTest 2: With dates and zones")
    # Note: evaluation_scoring might fail if data is too small or doesn't match expected levels,
    # but we want to check if it's called correctly.
    results = evaluate_metrics(y_true, y_pred, dates=dates, zones=zones)
    print(f"Score: {results['score']}")
    assert 'score' in results
    
    # Test 3: With y_pred_probas
    print("\nTest 3: With y_pred_probas")
    y_pred_probas = np.random.rand(6, 3)
    results = evaluate_metrics(y_true, y_pred, y_pred_probas=y_pred_probas)
    print(f"Entropy: {results['ent']}")
    assert results['ent'] > 0

    print("\nAll tests passed (or at least ran without crashing)!")

if __name__ == "__main__":
    try:
        test_evaluate_metrics()
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
