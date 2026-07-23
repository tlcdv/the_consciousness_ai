# Causal Emergence 2.0 (SVD heuristic) integration -- STATUS (2026-07)

**Status: implemented and wiring-verified. No scientific verdict yet.** This is an
instrument-integration record, not an experiment result. No causal-emergence claim
may be drawn from it until CE 2.0 is replicated across >= 3 seeds (42/43/44).

## What was built
Causal Emergence 2.0 (Hoel 2025, arXiv:2503.13395v3, Supplementary S3) as an
additive, default-off, diagnostic-only metric that supersedes Hoel's 2013 Effective
Information (EI / "CE 1.0"), which had a documented floor artifact
(`docs/results/instrument_repair_2026_07.md`).

- `models/evaluation/causal_emergence_svd.py` (new): the SVD heuristic on a
  row-stochastic TPM. `compute_ce2_from_tpm` discards the trivial sigma_1, takes
  `gamma* = mean(sigma_2..sigma_n)` (approximates determinism + specificity), returns
  `CE = sigma_2 - gamma*` and the emergent complexity = count of sigma_i (i>=2) above
  gamma*. No log2(n) size term. Plus the RSSM latent extractor
  (`latent_class_indices`, argmax over the categorical class axis) and memory-bounded
  incremental transition counting.
- `scripts/training/metrics_logger.py`: `compute_and_log_ce2` scores the gate (243),
  workspace (8), and RSSM class-transition (32x32) TPMs. CE 2.0 uses its own buffers,
  so EI's window is never altered. Seven columns added to `episodes.csv`.
- `scripts/training/train_rlhf.py`: `--log-ce2-every` (default 0), `--ce2-num-classes`
  (default 32). Per-step latent capture in `run_episode`, per-episode scoring in the
  main loop. RSSM CE 2.0 runs in `--rssm-latent-mode discrete` (the categorical path
  the instruction targets); continuous mode is a binned secondary path.

## Verification done this session
- Unit tests (`tests/test_causal_emergence_svd.py`, 17): analytic TPMs match hand
  derivation -- permutation -> CE 0; all-to-all uniform -> CE 0; k uniform blocks ->
  complexity k-1 (2 blocks -> 1, 3 -> 2, 4 -> 3); sigma_1 >= 1 for row-stochastic
  matrices; numerical-noise floor. Logger tests (`tests/test_metrics_logger.py`, +7).
- Full suite: 807 passed, 4 skipped, 0 failed.
- Reached-the-code-path check (dark_room, `--log-ce2-every 1`): `episodes.csv`
  populated `ce2_rssm` and `ce2_complexity_rssm` with finite non-zero values and no
  zero-transition warning, confirming the per-step capture and per-episode scoring
  both fired. These numbers are wiring evidence only, not a measurement.
- Bit-identical baseline: same seed (42), CE 2.0 off vs on, the per-step
  `metrics.csv` md5 is identical (`6ad1fce3...`). CE 2.0 consumes no RNG and runs
  under no_grad, so it does not perturb training. With the flag off, all seven new
  `episodes.csv` columns are exactly zero and existing EI columns are unchanged.

## Not done / owner decisions pending
- No >= 3-seed run, so no verdict on what CE 2.0 measures in this system.
- No CE 2.0 predictions pre-registered (see `docs/preregistered_predictions.md`).
- EI is deprecated in docs but retained (tests + pre-registration intact). Flipping
  the default off EI, and deleting the EI path, are deferred until CE 2.0 is
  validated.
- Pooled per-class TPM assumes the categorical variables share transition statistics;
  per-category averaging and the continuous-mode binned variant are untested here.
