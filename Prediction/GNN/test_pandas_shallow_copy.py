import pandas as pd
import numpy as np

def assert_not_changed(df1, df2, msg):
    try:
        pd.testing.assert_frame_equal(df1, df2)
        print(f"[PASS] {msg}")
    except AssertionError as e:
        print(f"[FAIL] {msg} - {e}")

# Setup
df = pd.DataFrame({
    'a': [1, 2, 3],
    'b': [4, 5, 6],
    'weight': [10, 20, 30]
})
print("Original DF:")
print(df)

# Test 1: deep=False
df_copy = df.copy(deep=False)

# Modification: Assign scalar to existing column
df_copy['weight'] = 0

print("\nModified Copy ('weight' = 0):")
print(df_copy)

print("\nOriginal DF after modification:")
print(df)

# Check if original is unchanged
expected_df = pd.DataFrame({
    'a': [1, 2, 3],
    'b': [4, 5, 6],
    'weight': [10, 20, 30]
})

assert_not_changed(df, expected_df, "Original 'weight' should remain [10, 20, 30]")

# Test 2: loc assignment (risky?)
df_copy2 = df.copy(deep=False)
try:
    df_copy2.loc[:, 'weight'] = 0
    print("\n[INFO] df_copy2.loc[:, 'weight'] = 0 executed")
    print("Original DF after loc modification:")
    print(df)
except Exception as e:
    print(f"[ERROR] with .loc: {e}")

# Check if original is changed (it SHOULD be changed if it's a view, or CoW not active)
# In many pandas versions, .loc[:, col] = val on a shallow copy modifies the original!
# But df['col'] = val replaces the column.

assert_not_changed(df, expected_df, "Original 'weight' should remain [10, 20, 30] after .loc on shallow copy")


# Test 3: Sequence used in code
print("\n--- Test 3: Sequence used in code ---")
df3 = pd.DataFrame({
    'a': [1, 2, 3],
    'b': [4, 5, 6],
    'weight': [10, 20, 30]
})
df_copy3 = df3.copy(deep=False)

# 1. Direct assignment (should break link)
df_copy3['weight'] = 0

# 2. Loc assignment on the copy (should modify only the copy's new column)
df_copy3.loc[0, 'weight'] = 1

print("df_copy3 'weight':")
print(df_copy3['weight'].values)

print("df3 'weight' (Original):")
print(df3['weight'].values)

assert_not_changed(df3, expected_df, "Original 'weight' should DEFINITELY remain [10, 20, 30] after sequence")
