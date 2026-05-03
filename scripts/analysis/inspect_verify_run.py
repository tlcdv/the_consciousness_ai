"""Inspect what is varying in a verification run.

Usage: python scripts/analysis/inspect_verify_run.py [run_dir]
Default run_dir: runs/verify_phi_fix
"""
import sys
import pandas as pd

run_dir = sys.argv[1] if len(sys.argv) > 1 else 'runs/verify_phi_fix'
df = pd.read_csv(f'{run_dir}/metrics.csv')
print('rows:', len(df))
print()
print('Variation per column:')
for col in ['phi', 'sync_r', 'is_conscious', 'reward', 'broadcast_mag',
            'valence', 'arousal', 'dominance']:
    print(f'  {col:15s}: nunique={df[col].nunique():4d} '
          f'mean={df[col].mean():.4f} std={df[col].std():.4f}')

print()
print('phi_method counts:', df['phi_method'].value_counts().to_dict())
print()
print('Sampled rows:')
for i in [0, 49, 99, 149, 199, 249]:
    if i < len(df):
        r = df.iloc[i]
        print(f'  step={i:3d} phi={r["phi"]:.3e} sync_r={r["sync_r"]:.4f} '
              f'bcast={r["broadcast_mag"]:.4f} method={r["phi_method"]}')
