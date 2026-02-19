
import re

file_path = '/home/caron/Bureau/ST-GNN-for-wildifre-prediction/Prediction/GNN/dataloader.py'

with open(file_path, 'r') as f:
    content = f.read()

wrappers = [
    'wrapped_train_deep_learning_1D_federated',
    'wrapped_train_deep_learning_1D_alafederated',
    'wrapped_train_deep_learning_1D_moonfederated',
    'wrapped_train_deep_learning_1D_federatedProx',
    'wrapped_train_deep_learning_1D_federatedfltg',
    'wrapped_train_deep_learning_1D_protofederated',
    'wrapped_train_deep_learning_1D_splittraining',
    'wrapped_train_deep_learning_1D_unique',
    'wrapped_train_deep_learning_1D_dualtraining',
    'wrapped_train_deep_learning_1D_distrib2classTraining',
    'wrapped_train_deep_learning_2D',
    'wrapped_train_deep_learning_2D_federated',
    'wrapped_train_deep_learning_distallation',
    'wrapped_train_deep_learning_hybrid',
    #'wrapped_train_sklearn_api_and_pytorch_voting_model' # This might need special handling
]

# Function to extract content lines
lines = content.split('\n')
new_lines = []

i = 0
in_wrapper = False
current_wrapper = ""

while i < len(lines):
    line = lines[i]
    
    # improved regex to match function definitions
    match = re.match(r'def (wrapped_train_\w+)\(', line)
    if match:
        wrapper_name = match.group(1)
        if wrapper_name in wrappers:
            in_wrapper = True
            current_wrapper = wrapper_name
            print(f"Processing {wrapper_name} at line {i+1}")
    
    # Check if we are inside a wrapper and need to inject extraction
    if in_wrapper:
        # Looking for end of params extraction to insert loss_param_search
        # Usually ends with n_run or c_n_run or similar
        # But we can just insert it after the extracted params block, or before split
        
        # A good anchor is "infos.split" or just check for 'infos = params['infos']' and insert after?
        # Actually, let's insert after 'n_run = ...' or 'c_n_run = ...'
        
        if "params['n_run']" in line or "params['client_n_run']" in line:
            # Check if loss_param_search is already extracted in next lines
            already_present = False
            for k in range(1, 10): # look ahead a bit
                if i+k < len(lines) and "loss_param_search =" in lines[i+k]:
                    already_present = True
                    break
            
            # If not present, we will insert it, but we need to wait until the last param extraction
            # Easier: Find the line "under_sampling, ... = infos.split" and insert BEFORE it
            pass

        if "infos.split('_')" in line:
             # Check if loss_param_search is already extracted above
            already_present = False
            for k in range(1, 20): # look back
                if i-k >= 0 and "loss_param_search =" in lines[i-k]:
                    already_present = True
                    break
            
            if not already_present:
                 # Insert before this line
                 indent = line[:len(line) - len(line.lstrip())]
                 new_lines.append(f"{indent}loss_param_search = params.get('loss_param_search', False)")
                 print(f"  Inserted extraction at line {i}")
                 
        # Look for Model instantiation to pass parameter
        # Pattern: horizon=int(horizon) followed by ) or ))
        
        if "horizon=int(horizon)" in line:
             if "loss_param_search=loss_param_search" in line:
                 pass # Already there (e.g. from previous attempts)
             else:
                 # Check if next line has loss_param_search (multiline call)
                 next_line = lines[i+1] if i+1 < len(lines) else ""
                 if "loss_param_search=" in next_line:
                     pass
                 else:
                     # We need to append it.
                     # Handle trailing comma and parentheses
                     stripped = line.rstrip()
                     indent = line[:len(line) - len(line.lstrip())]
                     
                     if stripped.endswith("))"):
                         # ModelGNN usually: horizon=int(horizon))
                         # Change to: horizon=int(horizon),
                         #            loss_param_search=loss_param_search)
                         new_line = line.replace("))", "),")
                         new_lines.append(new_line)
                         new_lines.append(f"{indent}loss_param_search=loss_param_search)")
                         print(f"  Updated ModelGNN call at line {i+1}")
                         i += 1
                         continue
                         
                     elif stripped.endswith(")"):
                         # Model_Torch usually: horizon=int(horizon) -> next line )
                         # OR horizon=int(horizon)) (double paren case covered above)
                         
                         # Check if it is single paren closing the call
                         # Usually Model_Torch call spans multiple lines and this is the last arg
                         
                         # We want to change "horizon=int(horizon)" to "horizon=int(horizon),"
                         # and add new line
                         
                         # Ensure we don't break simple calls
                         if stripped.endswith("),"):
                             # already has comma
                             new_lines.append(line)
                             new_lines.append(f"{indent}loss_param_search=loss_param_search,")
                             print(f"  Updated call (comma) at line {i+1}")
                             i += 1
                             continue
                         else:
                             # Replace ) with ), and append
                             new_line = line.replace(")", "),")
                             # Use regex to replace last )
                             # Actually, for Model_Torch: horizon=int(horizon)
                             # Next line is )
                             
                             # Let's peek next line
                             if i+1 < len(lines) and lines[i+1].strip() == ")":
                                  # This is the Multi-line usage
                                  new_lines.append(line + ",")
                                  new_lines.append(f"{indent}loss_param_search=loss_param_search")
                                  print(f"  Updated Model_Torch call at line {i+1}")
                                  i += 1
                                  continue # next loop will process the ) line
                                  
                             # One liner or end of line )
                             if stripped.endswith("id_past_ba = None"): # random check
                                 pass
                             
                             # Fallback: regex replace last ) with ),\n loss...
                             # verify strict matching
                             if line.strip() == "horizon=int(horizon)":
                                  new_lines.append(line + ",")
                                  new_lines.append(f"{indent}loss_param_search=loss_param_search")
                                  print(f"  Updated Model_Torch call (strict) at line {i+1}")
                                  i += 1
                                  continue

    new_lines.append(line)
    i += 1

with open(file_path, 'w') as f:
    f.write('\n'.join(new_lines))
