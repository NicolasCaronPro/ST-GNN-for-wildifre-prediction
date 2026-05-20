import pickle
import pandas as pd
import glob
files = glob.glob('/home/caron/Bureau/ST-GNN-for-wildifre-prediction/Prediction/GNN/firemen/firepoint/2x2/test/occurence_01_06_25/all/full_all_3_0_risk-size-zonemeteo-degree-a3-r5-t0.3_node/DilatedCNN_search_full_10_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_bceloss/H0/*_all_pred.pkl')
with open(files[0], 'rb') as f:
    data = pickle.load(f)
print("DFE unique values:", data.get('DFE').unique() if 'DFE' in data.columns else "Not found")
print("FWI summary:", data['FWI'].describe() if 'FWI' in data.columns else "Not found")
