# GPU checks after the code audit, 2026-10-03

Four checks that the audit of 2026-09-25 could not run, because they need a GPU or a
Python with pyphi. All ran on revision `d680aac`, Python 3.8.3, torch 2.4.1+cu121,
pyphi 1.2.0, one job at a time. Completion was read from csv row counts. No default and
no public claim is changed by this document.

## Verdicts first

1. **FAILED. `--broadcast-mode attention_weighted` cannot run on a GPU.** Both commands
   stop with a device mismatch before the first step is logged.
2. **FAILED. Gate B4 does not pass on the current code, with either merge rule.** With
   the default merge the gate fails at seed 57 and on the task criterion. With
   `--broadcast-merge top_winner` it fails at all three seeds. The published Gate B4
   values are correct for the original runs, and those runs cannot be reproduced from
   their seeds.
3. **No pyphi error in 1000 steps.** `phi_method` holds `pyphi` 199 times, `skipped`
   800 times and `insufficient_data` once. `pyphi_error` and `proxy` do not occur.
4. **No probe rerun crashed.** 28 reruns of 5 probes completed. 23 are identical to the
   original output files in every shared cell, and one more equals the table in its
   verdict document. No verdict is shown to rest on a substituted zero broadcast.

## 1. Device check for attention weighted fusion. FAILED

```
python -m scripts.training.train_rlhf --env dark_room --episodes 1 --max-steps 20 --existence-drive on --broadcast-mode attention_weighted --log-dir runs/audit_attn_gpu
python -m scripts.training.train_rlhf --env dark_room --episodes 1 --max-steps 20 --existence-drive on --broadcast-mode attention_weighted --enable-content-binding --log-dir runs/audit_bind_gpu
```

| Command | Error | Raised at | Source |
|---|---|---|---|
| fusion only | `RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!` | `models/core/global_workspace.py:564`, in `_fuse_tensor_payloads` | `runs/audit_attn_gpu.log`, line 18 |
| fusion with content binding | `RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cpu and cuda:0!` | `models/core/binding_attention.py:95`, called from `global_workspace.py:318` | `runs/audit_bind_gpu.log`, line 28 |

Each `metrics.csv` holds the header line and no step. The suspected CPU and CUDA mix is
confirmed. The mode is off by default, so no run with the default broadcast mode is
affected. Nothing here was repaired.

## 2. Gate B4 with `--broadcast-merge top_winner`. FAILED

### The original runs cannot be reproduced

A control came first. Seed 57 with the original Gate B4 flags and the default merge on
the current code gives a `metrics.csv` that differs from `runs/gate_b4_s57/metrics.csv`
in 16 of 39 columns, from step 0. A second run of the same command is bit identical to
the first (md5 `c06baf798c35f415fb14da6d5f379da3` for both), so the current code is
deterministic and the difference is not noise.

One cause is known. Commit `ab1bf7b` of 2026-09-25 seeds the environment's trial
generator from `--seed`. The original runs are from 2026-09-16, and their run facts
carry no `env_seeded` entry while the new runs carry `env_seeded: true`
(`session.json` in each run folder). Whether other changes since 2026-09-16 also
contribute was not measured.

Because of this, the new flag is compared with control runs on the same code, same
seeds and same flags. The published values are shown beside them.

### Result

Flags as in Gate B4 (`docs/results/dark_room_senses_2026_09.md`), seeds 57, 58, 59,
10 episodes and 2000 logged steps per run, 1990 steps judged. Each cell lists seed
57 / 58 / 59.

| Criterion | Published | Original runs, probed again | Default merge, current code | `top_winner`, current code |
|---|---|---|---|---|
| (1) runner-up share, at least 0.05 | 0.072 / 0.202 / 0.488 | 0.072 / 0.202 / 0.488 | **0.017** / 0.127 / 0.356 | **0.002 / 0.002 / 0.039** |
| (2) silence, below 0.50 | 0.133 / 0.247 / 0.322 | 0.133 / 0.247 / 0.322 | 0.104 / 0.239 / 0.307 | 0.043 / 0.124 / 0.247 |
| (3) selectivity, difference | 0.083 / 0.249 / 0.095 | 0.083 / 0.249 / 0.095 | 0.140 / 0.240 / 0.156 | 0.130 / 0.306 / 0.191 |
| (3) selectivity, null p95 | 0.011 / 0.159 / 0.017 | 0.011 / 0.159 / 0.017 | 0.072 / 0.160 / 0.058 | 0.025 / 0.211 / 0.087 |
| KILL, top share 0.95 or more | 0.928 / 0.798 / 0.512 | 0.928 / 0.798 / 0.512 | **0.983** / 0.873 / 0.644 | **0.998 / 0.998 / 0.961** |
| (4) task, rho per seed | 0.562 / 0.275 / 0.736 | 0.562 / 0.275 / 0.736 | 0.181 / 0.239 / 0.577 | 0.060 / -0.529 / 0.127 |
| (4) task, pooled rho and one-sided p | 0.458, 0.006 | 0.458, 0.006 | **0.289, 0.084** | **0.128, 0.632** |
| Audio share against episode index, rho | -0.194 / 0.512 / 0.815 | -0.194 / 0.512 / 0.815 | -0.149 / 0.460 / 0.450 | -0.522 / -0.199 / 0.693 |
| Probe verdict | PASS | PASS | FAILED | FAILED |

Bold marks a failed criterion. Probe failure lines, shortened from the outputs.

- Default merge. `runs/gate_b4_legacy_ctrl_s57: KILL top share 0.983; (1) competition; (4) task`.
- `top_winner`. KILL and (1) at seeds 57, 58 and 59, (3) selectivity at seed 57
  (silence 0.043 is below the 0.05 minimum), and (4) task.

In every run the top module is vision, except the original seed 59 where audio holds
0.512.

Sources. `runs/gate_b4_s57..59` (original), `runs/gate_b4_legacy_ctrl_s57..59`,
`runs/gate_b4_top_winner_s57..59`, each judged by
`python -m scripts.analysis.probe_gate_b2 --runs <three folders>`.

### What this establishes and what it does not

- The published Gate B4 numbers are reproduced exactly from the original run folders.
- The same seeds and flags on the current code do not pass the gate. The Gate B4 pass
  therefore holds for the trials those three runs received, and it is not a property
  that survives a change of trials at the same seeds.
- At these three seeds `top_winner` gives vision 0.961 or more of ignited steps and
  removes the task link. The flag changes which tensor the policy reads when two
  modules win, so it changes behaviour, and the comparison with the control isolates
  that change.
- Not established. Whether the environment seeding is the only cause of the difference
  from the original runs. Whether other seeds would pass. Three seeds were run per arm
  and no further seeds.
- The default of `--broadcast-merge` is unchanged. The public Gate B4 text is unchanged.
  Both are decisions for the project owner.

## 3. pyphi error count

```
python -m scripts.training.train_rlhf --env dmts --episodes 5 --max-steps 200 --seed 42 --existence-drive on --log-dir runs/audit_phi_errors
```

`runs/audit_phi_errors/metrics.csv` holds 1000 steps and `episodes.csv` holds 5
episodes. The run writes one `metrics.csv`.

| `phi_method` | Steps |
|---|---|
| `skipped` | 800 |
| `pyphi` | 199 |
| `insufficient_data` | 1 |
| `pyphi_error` | 0 |
| `proxy` | 0 |

Phi on the 199 `pyphi` steps ranges from 0.000001 to 0.003802. One seed, one
environment and one configuration, so this is a count for this run and not a rate.

## 4. Probe reruns

Before the audit, a failure inside `_compute_broadcast` was replaced with zeros. The
function now raises. Each probe that uses it was run again with the checkpoint and
arguments recorded in its verdict document, with the output written to a new folder
(`runs/audit_rerun/`) so that no original file was overwritten. A crash would mean the
earlier output held substituted zeros.

| Probe | Verdict documents | Reruns | Crashes | Compared with the original output |
|---|---|---|---|---|
| `probe_pci` | `pci_trained_2026_08`, `pci_gate_saturation_2026_09`, `pci_multi_checkpoint_2026_09`, `pci_three_trained_gates_2026_09`, `pci_reading_rules_2026_09`, `pci_probe_seed_null_2026_09`, `pci_content_2026_09` | 18 | 0 | 18 of 18 identical in every shared cell. Newer files carry up to 3 added columns |
| `probe_gate_attenuation` | `pci_gate_attenuation_2026_09` | 5 | 0 | 5 of 5 identical |
| `probe_interventional_tpm` | `interventional_tpm_2026_08` | 1 | 0 | No original file. The printed table equals the table in the verdict document (0.021826, 0.020412, 0.758312, 0.002272, 0.002201, 0.016380, 56 states) |
| `probe_workspace_ordering` | `thalamic_instruments_2026_07` | 2 | 0 | Differs, see below |
| `probe_perception_decodability` | `perception_decodability_2026_06_09` | 2 | 0 | Differs in `test_acc`, see below |

**Verdicts affected by a crash. None.**

Two reruns do not match their documents, and the cause is not the zero substitution.

- **`probe_workspace_ordering`.** The verdict document records no command line. The
  reruns used `--env dmts --seed 42` with the default configuration and with
  `--latent-mode continuous --capsule-workspace-source all_levels`. Signal standard
  deviations, document against rerun. Default configuration, `sync_R` 9.765e-03
  against 7.975e-03 and broadcast 7.132e-11 against 7.095e-11. Fixed configuration,
  broadcast 5.472e-02 against 4.858e-02, `sync_R` 7.337e-03 against 5.488e-03,
  `gate_coherence` 1.098e-04 (usable) against 9.750e-05 (not usable). The original
  arguments are unknown, so the two sides are not a like for like comparison.
- **`probe_perception_decodability`.** Sample counts, chance levels and majority
  baselines are identical in all 54 shared rows. `test_acc` differs in 8 of 36 shared
  rows of the untrained sweep (largest difference 0.05) and in 6 of 18 shared rows of
  the trained probe (largest difference 0.1). Every differing row is the
  `tectum_content` stage or a `broadcast` row with the same value as its
  `tectum_content` row. The trained probe ran with `--no-broadcast` and never calls
  `_compute_broadcast`, and it differs as well. The cause was not identified. No
  conclusion of that verdict document was re-judged here.

Four further recorded commands of `probe_perception_decodability`, in
`rssm_working_memory_2026_06_12` and `tectum_reconstruction_2026_06_10`, use
`--no-broadcast` and do not reach `_compute_broadcast`. They were not rerun.

Sources. `runs/audit_rerun/*.csv`, `runs/audit_rerun/*.full.txt`, and the original
files under `runs/pci_trained/`, `runs/_pci_multi/`, `runs/_pci_null/`,
`runs/_pci_content/`, `runs/perception_probe_full/` and `runs/perception_probe_trained/`.

## Open decisions for the project owner

1. Repair or retire `--broadcast-mode attention_weighted`, which cannot run on a GPU.
2. How to state Gate B4 in public text, given that it does not pass on the current
   code at the same seeds. A rerun on fresh seeds with a gate written before the runs
   is the usual next step.
3. The default of `--broadcast-merge`. At these three seeds `top_winner` fails the gate
   more strongly than the default merge.
4. Whether to find the cause of the `test_acc` differences in the decodability probe.
