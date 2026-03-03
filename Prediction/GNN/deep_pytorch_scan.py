import os
import re

anti_patterns = {
    'Missing .item() on loss append': r'\.append\([^)]*loss[^)]*\)',
    'Missing .detach() on output/pred append': r'\.append\([^)]*(?:out|pred|hidden|logits)[^)]*\)',
    'Missing .item() in dict assignment': r'\[.*\]\s*=\s*[^=]*loss',
    'Direct model assignment': r'^\s*(?!self\.)[a-zA-Z0-9_]+\s*=\s*(?:self\.)?model\b',
    'Direct parameter assignment': r'^\s*[a-zA-Z0-9_]+\s*=\s*.*\.parameters\(\)',
    'Direct weight/bias assignment': r'^\s*[a-zA-Z0-9_]+\s*=\s*.*\.(?:weight|bias)\b'
}

safe_keywords = ['item()', 'detach()', 'clone()', 'deepcopy', 'cpu()', 'numpy()', 'float(']

def scan_file(filepath):
    results = []
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        
    for i, line in enumerate(lines):
        line = line.strip()
        if not line or line.startswith('#'): continue
        
        for name, pattern in anti_patterns.items():
            if re.search(pattern, line, re.IGNORECASE):
                # If it's a loss or tensor operation, check for safe keywords
                if any(k in line for k in safe_keywords):
                    continue
                
                # Exclude obvious non-PyTorch things
                if 'loss_params' in line or 'patience' in line or 'loss_res' in line:
                    continue
                if 'append(loss_dict' in line or 'append(train_loss' in line or 'append(val_loss' in line:
                    # Often float already, but let's flag if it looks like raw loss
                    pass
                
                # We flag this
                results.append((i+1, name, line))
                
    return results

total_issues = 0
for root, _, files in os.walk('.'):
    # skip baseline, env, etc
    if 'baseline' in root or '.git' in root or '__pycache__' in root:
        continue
    for fl in files:
        if fl.endswith('.py'):
            filepath = os.path.join(root, fl)
            issues = scan_file(filepath)
            if issues:
                print(f"--- {filepath} ---")
                for line_num, name, content in issues:
                    if len(content) > 100: content = content[:100] + '...'
                    print(f"  Line {line_num}: [{name}] {content}")
                    total_issues += 1

print(f"\nTotal potential issues found: {total_issues}")
