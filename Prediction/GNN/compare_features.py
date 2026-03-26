import ast

def extract_X_assignments(filepath, func_name):
    with open(filepath, 'r') as f:
        tree = ast.parse(f.read())
    
    func_node = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            func_node = node
            break
            
    if not func_node: return []
    
    assignments = []
    for node in ast.walk(func_node):
        if isinstance(node, ast.Assign) or isinstance(node, ast.AnnAssign):
            # Check targets
            targets = getattr(node, 'targets', [getattr(node, 'target', None)])
            for target in targets:
                if isinstance(target, ast.Subscript):
                    if isinstance(target.value, ast.Name) and target.value.id == 'X':
                        # This is an assignment to X[...]
                        # Let's try to unparse the slice
                        try:
                            
                            slice_str = ast.unparse(target.slice)
                            val_str = ast.unparse(node.value) if node.value else ""
                            assignments.append(f"X[{slice_str}] = {val_str}")
                        except:
                            assignments.append(f"X assignment found at line {node.lineno}")
    return assignments

features_file = '/home/caron/Bureau/ST-GNN-for-wildifre-prediction/Prediction/GNN/features.py'

print("--- XARRAY ASSIGNMENTS ---")
xarray_ass = extract_X_assignments(features_file, 'get_sub_nodes_features_from_xarray')
for a in xarray_ass: print(a)

print("\n--- GEODATAFRAME ASSIGNMENTS ---")
df_ass = extract_X_assignments(features_file, 'get_sub_nodes_feature_with_geodataframe')
for a in df_ass: print(a)
