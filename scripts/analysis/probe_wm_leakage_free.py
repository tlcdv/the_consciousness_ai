"""
Leakage-free working-memory probe for DMTS.

This is the corrective companion to probe_perception_decodability.py. That probe
decodes the RSSM recurrent state `h_state` per step with a random train/test split,
which LEAKS for temporally-correlated states: consecutive delay frames within a
trial are near-identical and carry the same sample label, so the decoder memorizes
trials rather than generalizing. That leakage produced the false 2026-06-12 claim
that `h_state` holds the sample at ~99% across the delay
(docs/results/rssm_working_memory_2026_06_12.md CORRECTION).

This script collects exactly ONE record per trial for each representation, so no two
records in a decode come from the same trial. It reports:

  - sample obs_map  (control: stimulus on screen, should decode high)
  - sample h_state  (does the RSSM encode the sample while it is visible?)
  - delay  h_state  (does the RSSM retain the sample across the blank delay?)

Leakage-free result (seed 42, 12 episodes, n=240): obs_map ~0.97-1.00 (real),
h_state at chance at both sample and delay. The RSSM does not build a usable working
memory of the sample; a working-memory capability has to be BUILT, not merely wired.

Run:
    PYPHI_WELCOME_OFF=yes python -m scripts.analysis.probe_wm_leakage_free \
        --episodes 12 --seed 42
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from scripts.analysis.probe_perception_decodability import (
    _build_components, frame_to_tensor, linear_decode,
)
from simulations.environments.dmts_env import DMTSEnv
from models.self_model.working_memory_latch import ObsMapSampleMemory


def collect(episodes: int, seed: int, load_tectum=None, wm_action_dim: int = 0):
    cfg, tectum, *_ = _build_components("dmts", action_dim=5, seed=seed,
                                        mock_semantic=False, load_tectum=load_tectum,
                                        wm_action_dim=wm_action_dim)

    def _onehot(a):
        # Reproduce the action-conditioned RSSM dynamics of a wm-predict checkpoint.
        if wm_action_dim <= 0 or a is None:
            return None
        oh = torch.zeros(1, wm_action_dim, device=cfg["device"])
        if 0 <= int(a) < wm_action_dim:
            oh[0, int(a)] = 1.0
        return oh

    env = DMTSEnv(num_trials=20)
    # Gated obs_map memory: captures the sample obs_map (after the short fixation
    # blank) and holds it through the delay/choice. mem_slot is recorded at the
    # choice phase to test whether the held sample is available at the decision.
    mem = ObsMapSampleMemory()
    samp_obs, samp_h, delay_h, mem_slot = [], [], [], []
    ys = {"shape": [], "color": []}
    rng = np.random.default_rng(seed)
    with torch.no_grad():
        for ep in range(episodes):
            obs, info = env.reset(seed=seed + ep)
            tectum.reset_state(1)
            mem.reset()
            s_obs = s_h = d_h = lab = None
            prev_a = None
            done, steps = False, 0
            while not done and steps < 4000:
                ph = info.get("phase")
                f = frame_to_tensor(obs, cfg["device"])
                tectum(f, torch.zeros(1, cfg["tectum_feature_dim"], 2,
                                      device=cfg["device"]), action=_onehot(prev_a))
                slot = mem.update(tectum._last_obs_map, f)
                if ph == "sample":
                    s_obs = tectum._last_obs_map.reshape(-1).cpu().numpy().astype(np.float32)
                    s_h = tectum.h_state.reshape(-1).cpu().numpy().astype(np.float32)
                    lab = (info["sample_shape"], info["sample_color"])
                elif ph == "delay":
                    d_h = tectum.h_state.reshape(-1).cpu().numpy().astype(np.float32)
                elif ph == "choice" and s_obs is not None and d_h is not None:
                    # one record per trial: no within-trial leakage
                    samp_obs.append(s_obs); samp_h.append(s_h); delay_h.append(d_h)
                    mem_slot.append(slot.reshape(-1).cpu().numpy().astype(np.float32))
                    ys["shape"].append(lab[0]); ys["color"].append(lab[1])
                    s_obs = s_h = d_h = lab = None
                a = 0 if ph != "choice" else int(rng.integers(1, env.num_choices + 1))
                prev_a = a
                obs, _, term, trunc, info = env.step(a)
                done = term or trunc
                steps += 1
    return samp_obs, samp_h, delay_h, mem_slot, ys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--load-tectum", type=str, default=None,
                    help="Load a trained tectum state_dict before probing.")
    ap.add_argument("--wm-action-dim", type=int, default=0,
                    help="Set to the env action dim (e.g. 5 for DMTS) when loading a "
                         "--enable-wm-predict checkpoint, so the action embedding loads "
                         "and the probe reproduces the action-conditioned RSSM dynamics.")
    args = ap.parse_args()

    mode = f"TRAINED ({args.load_tectum})" if args.load_tectum else "UNTRAINED init"
    print(f"probe mode: {mode}  wm_action_dim={args.wm_action_dim}")
    samp_obs, samp_h, delay_h, mem_slot, ys = collect(
        args.episodes, args.seed, args.load_tectum, args.wm_action_dim)
    print(f"LEAKAGE-FREE one-per-trial decode, n={len(samp_obs)} trials")
    # sample obs_map: on-screen control (should be high). h_state: RSSM does not
    # retain the sample (chance). mem_slot: the gated obs_map memory at the CHOICE
    # phase (should hold the sample, the working-memory mechanism that works).
    for name, X in [("sample obs_map", samp_obs), ("sample h_state", samp_h),
                    ("delay h_state", delay_h), ("mem_slot @choice", mem_slot)]:
        for lab in ["shape", "color"]:
            r = linear_decode(X, ys[lab], seed=args.seed)
            print(f"  {name:16s} {lab:6s} acc={r['test_acc']:.3f} "
                  f"chance={r['uniform_chance']:.3f} major={r['majority']:.3f}")


if __name__ == "__main__":
    main()
