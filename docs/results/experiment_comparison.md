# Experiment Comparison: Consciousness Agent vs DQN Baseline

## Reward Comparison

| Environment | C Episodes | C First 100 | C Last 100 | D Episodes | D First 100 | D Last 100 |
|---|---|---|---|---|---|---|
| Dark Room | 492 | 9.40 | 12.95 | 1000 | 52.72 | 92.00 |
| DMTS | 100 | -9.82 | -9.82 | 500 | -28.85 | -4.07 |
| WCST | 100 | -1.94 | -1.94 | 500 | 1.05 | 2.06 |

## Consciousness Metrics

| Environment | Avg Phi | Phi Varies | EI Ratio | EI Measurements |
|---|---|---|---|---|
| Dark Room | 0.02201 | True | 2.414 | 9 |
| DMTS | 0.02202 | True | 2.418 | 4 |
| WCST | 0.02202 | True | 2.418 | 4 |

Note, 2026-10-03. This file was regenerated from `runs/dark_room_1k_v2`, `runs/dmts_100`, `runs/wcst_100` and `runs_baseline/`. The tables did not change. The earlier Findings section was removed because the script wrote it as fixed text and never computed it.
