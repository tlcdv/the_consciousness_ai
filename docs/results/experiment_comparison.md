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

## Findings

### Structural fixes applied (2026-03-29)

1. **ConsciousnessGate wired**: all 5 gate values (attention, stability, adaptation, coherence, confidence) computed from broadcast via learned networks. No longer static. Phi now varies per step.
2. **`compute_phi_proxy()` replaced** with `compute_phi_from_gate_state()` in GlobalWorkspace and training loop.
3. **Adaptive EI binning**: per-dimension median thresholds instead of fixed 0.5, so adaptation_rate (range 0.004-0.006) contributes to joint state diversity.
4. **DMTS/WCST action discretization**: consciousness agent now correctly converts continuous actions to discrete indices via argmax.

### Known limitations for this run

- Phi proxy converges to empirical fixed point after ~5000 steps (TPM saturates). Per-episode phi becomes constant after early training.
- EI stable across measurement windows: gate transitions converge to a stationary distribution quickly. Longer training or sliding-window TPM needed.
- DQN outperforms consciousness agent on all reward metrics. The consciousness pipeline adds ~200ms overhead per step without contributing to the action policy directly.
- sync_R range [0.216, 0.220]: workspace binding optimizer needs stronger reward signal over many more episodes to shift coupling weights significantly.
