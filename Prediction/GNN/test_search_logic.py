import numpy as np
import math

class MockModel:
    def __init__(self):
        self.metrics = {}
        self.dir_log = '.'
        self.reference_scores = None
    
    def _compute_geometric_agg(self, mapped_dict):
        # Dummy agg: just the average of scores
        vals = [v for v in mapped_dict.values() if isinstance(v, (int, float))]
        return np.mean(vals), None

    def _mean_u_agg(self, m_dict, suffix='_val'):
        mapped_dict = {}
        for k in [1, 2, 3, 4]:
            vals = np.atleast_1d(m_dict.get(f'score_k{k}{suffix}', [0.0]))
            sk = float(np.nanmean(vals)) if len(vals) else 0.0
            mapped_dict[f'score_k{k}'] = sk if not np.isnan(sk) else 0.0
        
        rv = np.atleast_1d(m_dict.get(f'recall{suffix}', [0.0]))
        sk_r = float(np.nanmean(rv)) if len(rv) else 0.0
        mapped_dict['recall'] = sk_r if not np.isnan(sk_r) else 0.0
        
        smcv = np.atleast_1d(m_dict.get(f'score_min_class{suffix}', [0.0]))
        sk_smc = float(np.nanmean(smcv)) if len(smcv) else 0.0
        mapped_dict['score_min_class'] = sk_smc if not np.isnan(sk_smc) else 0.0
        
        agg, _ = self._compute_geometric_agg(mapped_dict)
        return float(agg)

def test_logic(metrics_data, use_log, find_log):
    model = MockModel()
    model.metrics = metrics_data
    data_log = metrics_data
    # In reality test_percentage is np.asarray from metrics
    test_percentage = np.asarray(data_log.get('test_percentage', [0.1, 0.2, 0.3, 0.4]))
    
    tolerance = 0.03
    doSearch = False
    last_score = -math.inf
    start_test = 0

    if find_log and use_log:
        if data_log is not None and 'test_percentage' in data_log:
            model.metrics = data_log
            test_percentage = np.asarray(model.metrics['test_percentage'])
            
            doSearch = True
            for i, tp_val in enumerate(test_percentage):
                tp_val = round(tp_val, 2)
                if tp_val in model.metrics:
                    current_agg = model._mean_u_agg(model.metrics[tp_val], '_val')
                    if current_agg >= last_score - tolerance:
                        if current_agg > last_score:
                            last_score = current_agg
                    else:
                        print(f'Stopping search: scores declining in logs (last_score={last_score:.4f}, current_agg={current_agg:.4f})')
                        doSearch = False
                        break
                else:
                    start_test = i
                    doSearch = True
                    print(f'Resuming search from tp={tp_val} (first missing in data_log, index {i})')
                    break
            else:
                doSearch = False
                print(f'All test_percentage values found in data_log → doSearch=False')
    else:
        doSearch = False
        if not use_log:
            print("Search disabled (use_log=False)")
        elif not find_log:
            print("Search disabled (No logs found and use_log=True)")
    
    return doSearch, start_test, last_score

# Case 1: Continuing search (good scores)
metrics_1 = {
    'test_percentage': [0.1, 0.2, 0.3, 0.4],
    0.1: {'score_k1_val': 0.5, 'score_k2_val': 0.5, 'score_k3_val': 0.5, 'score_k4_val': 0.5, 'recall_val': 0.5, 'score_min_class_val': 0.5},
    0.2: {'score_k1_val': 0.6, 'score_k2_val': 0.6, 'score_k3_val': 0.6, 'score_k4_val': 0.6, 'recall_val': 0.6, 'score_min_class_val': 0.6}
}
print("--- Case 1: Resuming search ---")
ds, st, ls = test_logic(metrics_1, use_log=True, find_log=True)
print(f"doSearch={ds}, start_test={st}, last_score={ls}")

# Case 2: Declining scores in logs
metrics_2 = {
    'test_percentage': [0.1, 0.2, 0.3, 0.4],
    0.1: {'score_k1_val': 0.5, 'score_k2_val': 0.5, 'score_k3_val': 0.5, 'score_k4_val': 0.5, 'recall_val': 0.5, 'score_min_class_val': 0.5},
    0.2: {'score_k1_val': 0.4, 'score_k2_val': 0.4, 'score_k3_val': 0.4, 'score_k4_val': 0.4, 'recall_val': 0.4, 'score_min_class_val': 0.4}
}
print("\n--- Case 2: Declining scores ---")
ds, st, ls = test_logic(metrics_2, use_log=True, find_log=True)
print(f"doSearch={ds}, start_test={st}, last_score={ls}")

# Case 3: All scores present and good
metrics_3 = {
    'test_percentage': [0.1, 0.2],
    0.1: {'score_k1_val': 0.5, 'score_k2_val': 0.5, 'score_k3_val': 0.5, 'score_k4_val': 0.5, 'recall_val': 0.5, 'score_min_class_val': 0.5},
    0.2: {'score_k1_val': 0.6, 'score_k2_val': 0.6, 'score_k3_val': 0.6, 'score_k4_val': 0.6, 'recall_val': 0.6, 'score_min_class_val': 0.6}
}
print("\n--- Case 3: All present ---")
ds, st, ls = test_logic(metrics_3, use_log=True, find_log=True)
print(f"doSearch={ds}, start_test={st}, last_score={ls}")

# Case 4: No logs
print("\n--- Case 4: No logs ---")
ds, st, ls = test_logic({}, use_log=True, find_log=False)
print(f"doSearch={ds}, start_test={st}, last_score={ls}")
