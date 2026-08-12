"""
Does the pre-sum `kl_map` carry the phase information that summing destroys?

`sensory_tectum.py:443-444` builds a 262,144-element `kl_map`, sums it to one scalar, and
squashes that scalar into the vision bid. The workspace-competition probe measured the
scalar and found it does not separate DMTS sample from delay (Cohen's d -0.061, -0.293,
-0.078 at 3 seeds). Summing 262,144 elements cancels structure, so the map may carry
phase information the scalar cannot.

This decides which of two very different repairs is even possible:

  - The map carries phase information. The bid FORMULA is wrong, a pooled or weighted bid
    could produce genuine selective attention, and GWT-2 becomes a repair.
  - The map does not. No bid formula helps, the loss is upstream in the representation,
    and GWT-1 / GWT-2 become a re-score decision rather than a repair.

PRE-STATED GATE, written before any number from this probe was read. Primary test is
sample versus delay on B1, because that is the discrimination DMTS requires and the one
that would give GWT-2 selective attention.

    REPAIRABLE BY BID FORMULA:
        B1 beats its own shuffled control D by >= 0.10 AND beats the majority-class
        baseline by >= 0.10, at all 3 seeds.
    NOT REPAIRABLE BY BID FORMULA:
        that margin is < 0.10 at any seed. No linear function of `kl_map` separates the
        phases, so the sum is not what loses the information.
    INCONCLUSIVE:
        the shuffled control D itself beats majority by >= 0.10. The probe is overfitting
        at 1024 features and no number in the run is readable.

Feature sets. `kl_map` is (B, 32, 32, 16, 16): 1024 channels by a 16x16 spatial grid.

    A   kl_map.sum(), the current bid input             1 feature
    B1  mean over the 16x16 grid                     1024 features
    B2  mean over the 1024 channels                   256 features
    C   _last_obs_map pooled over its grid              64 features   UPPER REFERENCE
    D   B1 with SHUFFLED labels                      1024 features   NOISE FLOOR
    T   step index within the trial                     1 feature    CLOCK CONFOUND

THE CLOCK CONFOUND, and why T and the identity test exist. In DMTS the phases run in a
fixed order with `fixation_steps=10` and `sample_steps=20`, so the sample-to-delay
boundary sits at a FIXED step index of 30 within every trial. Sample versus delay is
therefore a deterministic function of elapsed time, and ANY feature that drifts with time
decodes it perfectly without carrying one bit about task content. The RSSM is recurrent,
so `kl_map` drifting with time since reset is exactly what one would expect.

T measures how large that confound is. It is not a control that can be passed: if T
decodes near 1.0, the phase label is a clock reading and no phase decode on time
correlated features can separate content from clock.

The clock-free test is stimulus identity. `sample_shape` varies across trials and is
independent of step index, so decoding it from `kl_map` during the sample phase asks
whether the map carries task CONTENT. The collapse-locus work established that identity
dies at the RSSM, but it measured `z_state`, `capsule_poses` and `tectum_content`, never
`kl_map`. This is a new question.

A is contained in B1's hypothesis space, since equal weights reproduce a sum. So
acc(B1) >= acc(A) up to estimation noise, and the gap between them is exactly the
information the sum destroys.

C is read on a rule fixed in advance. C decoding while B1 does not means the information
reaches the tectum and the RSSM prediction error discards it, which points at the RSSM
and not at the bid. C also failing means the observation does not distinguish the phases
in this configuration, which points at the environment rendering.

LIMITATION, stated up front: a linear probe failing shows that no LINEAR function of
`kl_map` separates the phases. It does not prove no function does.

Read-only. No training run, no checkpoint written, no model modified.

Run:
    python -m scripts.analysis.probe_klmap_phase_information \\
        --load-tectum runs/gate_ckpt_s42/tectum.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.analysis.probe_perception_decodability import (
    _build_components, linear_decode,
)
from scripts.training.train_rlhf import frame_to_tensor
from simulations.environments.dmts_env import DMTSEnv

SEEDS = (42, 43, 44)
MARGIN = 0.10               # pre-stated
PRIMARY = ("sample", "delay")
PHASES = ("fixation", "sample", "delay", "choice")
FIDELITY_RTOL = 1e-4        # float32 sum over 262144 elements, recomputed independently


def collect(config, tectum, episodes: int, seed: int, max_steps: int = 200) -> dict:
    """Pool `kl_map` online, per step, and keep only the pooled vectors.

    Storing the raw map would be 262144 floats x 6000 steps, about 6 GB. Pooling online
    keeps it to 1344 floats per step.

    `kl_map` is recomputed from the tectum's own cached logits, which reproduces
    `sensory_tectum.py:443` exactly rather than approximating it. The fidelity check in
    main() gates every number in this run on that reproduction being exact.
    """
    pre_tanh: list[float] = []
    real_tanh = torch.tanh

    def tanh_spy(x, *a, **kw):
        # sensory_tectum.py:456 squashes a 0-dim scalar; the GRU candidate tanh at line
        # 239 is multi-element, so numel() separates them.
        if isinstance(x, torch.Tensor) and x.numel() == 1:
            pre_tanh.append(float(x.item()))
        return real_tanh(x, *a, **kw)

    A, B1, B2, C, C2, phases, recomputed = [], [], [], [], [], [], []
    step_in_trial, shapes, episode_of, trial_of = [], [], [], []
    torch.tanh = tanh_spy
    try:
        env = DMTSEnv(num_trials=20)
        with torch.no_grad():
            for ep in range(episodes):
                obs, info = env.reset(seed=seed + ep)
                if hasattr(tectum, "reset_state"):
                    tectum.reset_state(1)
                done, steps = False, 0
                cur_trial, in_trial = info.get("trial"), 0
                while not done and steps < max_steps:
                    frame = frame_to_tensor(obs, config["device"])
                    audio = torch.zeros(1, config["tectum_feature_dim"], 2,
                                        device=config["device"])
                    tectum(frame, audio)

                    post = tectum._last_post_logits
                    prior = tectum._last_prior_logits
                    var = torch.exp(tectum.rssm.cont_logvar)
                    kl_map = 0.5 * (post - prior) ** 2 / (var + 1e-8)

                    recomputed.append(float(kl_map.sum() / post.shape[0]))
                    A.append([float(kl_map.sum())])
                    # (B, 32, 32, H, W): pool the spatial grid -> per-channel vector.
                    B1.append(kl_map.mean(dim=(3, 4)).reshape(-1).cpu().numpy())
                    # pool the channels -> per-position vector.
                    B2.append(kl_map.mean(dim=(1, 2)).reshape(-1).cpu().numpy())
                    om = tectum._last_obs_map            # (B, 64, 16, 16)
                    C.append(om.mean(dim=(2, 3)).reshape(-1).cpu().numpy())
                    # Channel-pooled, SPACE PRESERVED. Stimulus shape is a spatial
                    # property, so a reference that pools the grid away cannot serve as
                    # a ceiling for the identity test. C2 is that ceiling.
                    C2.append(om.mean(dim=1).reshape(-1).cpu().numpy())
                    phases.append(info.get("phase"))
                    step_in_trial.append([float(in_trial)])
                    shapes.append(info.get("sample_shape"))
                    episode_of.append(ep)
                    trial_of.append((ep, info.get("trial")))

                    obs, _, term, trunc, info = env.step(0)
                    done = term or trunc
                    steps += 1
                    if info.get("trial") != cur_trial:
                        cur_trial, in_trial = info.get("trial"), 0
                    else:
                        in_trial += 1
    finally:
        torch.tanh = real_tanh

    return {
        "A": np.asarray(A, dtype=np.float64),
        "B1": np.asarray(B1, dtype=np.float64),
        "B2": np.asarray(B2, dtype=np.float64),
        "C": np.asarray(C, dtype=np.float64),
        "C2": np.asarray(C2, dtype=np.float64),
        "T": np.asarray(step_in_trial, dtype=np.float64),
        "phase": np.asarray(phases),
        "shape": np.asarray(shapes),
        "episode": np.asarray(episode_of),
        "trial": np.asarray([f"{e}:{t}" for e, t in trial_of]),
        "recomputed_kl": np.asarray(recomputed, dtype=np.float64),
        "spy_kl": np.asarray(pre_tanh[:len(recomputed)], dtype=np.float64),
    }


def fidelity(data: dict) -> tuple[bool, float]:
    """Recomputed kl_div must match the value the tectum actually squashed."""
    a, b = data["recomputed_kl"], data["spy_kl"]
    if a.size == 0 or a.size != b.size:
        return False, float("nan")
    denom = np.maximum(np.abs(b), 1e-12)
    worst = float(np.max(np.abs(a - b) / denom))
    return bool(worst <= FIDELITY_RTOL), worst


def decode(X, y, seed: int) -> dict:
    return linear_decode(X, y, seed=seed)


def grouped_decode(X, y, groups, seed: int = 0, test_frac: float = 0.3) -> dict:
    """Same protocol as `linear_decode`, but the split holds out WHOLE EPISODES.

    Why this exists. `linear_decode` splits at random over steps. Consecutive steps
    inside one DMTS phase are near-duplicates (the screen does not change during a
    delay), and `sample_shape` is constant for all ~20 steps of a trial. A random split
    therefore puts a near-twin of almost every test sample into the training set, and a
    classifier can score high by memorizing rather than generalizing.

    The shuffled-label control does NOT catch this. Shuffling gives near-twins DIFFERENT
    labels, so memorizing stops helping; with real labels the twins share a label and it
    helps. The control and the leak point in opposite directions.

    Two grouping units are reported, and they are not interchangeable.

    BY TRIAL is the minimal correct fix. The leak is within-trial duplication, so holding
    out whole trials removes exactly that and nothing else.

    BY EPISODE is strictly harder. Each episode runs a different env seed, so it also
    demands generalization across stimulus sequences. A score below chance under episode
    grouping indicates distribution shift between train and test episodes rather than
    absence of information, which is why both are shown rather than one.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    groups = np.asarray(groups)
    classes, y_idx = np.unique(y, return_inverse=True)
    n_classes, n = int(len(classes)), int(len(y_idx))
    majority = float(np.bincount(y_idx).max() / n) if n > 0 else float("nan")

    # Test groups are chosen at RANDOM, not as the tail of the sorted group list.
    # Taking the tail was a bug: np.unique sorts "episode:trial" strings
    # lexicographically, so the last 30% of trials are exactly the trials of the last
    # episodes, which silently turned trial grouping into episode grouping and made the
    # two report identical numbers. Random selection also keeps this a test of
    # generalization rather than of temporal extrapolation.
    uniq = np.unique(groups)
    n_test = max(1, int(round(len(uniq) * test_frac)))
    rng = np.random.default_rng(seed)
    test_groups = set(rng.choice(uniq, size=n_test, replace=False).tolist())
    te = np.array([g in test_groups for g in groups])
    tr = ~te

    if (n_classes < 2 or n < 10 or tr.sum() < 10 or te.sum() < 10
            or len(np.unique(y_idx[tr])) < 2):
        return {"test_acc": float("nan"), "majority": majority, "n": n,
                "n_classes": n_classes, "n_test": int(te.sum()), "method": "skip"}

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    clf.fit(X[tr], y_idx[tr])
    # Majority baseline on the held-out episodes, which is the number the accuracy
    # must beat. The all-data majority can differ from it.
    te_major = float(np.bincount(y_idx[te]).max() / te.sum())
    return {"test_acc": float(clf.score(X[te], y_idx[te])), "majority": te_major,
            "n": n, "n_classes": n_classes, "n_test": int(te.sum()),
            "method": "grouped"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-tectum", default="runs/gate_ckpt_s42/tectum.pt")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--latent-mode", default="continuous",
                        choices=["discrete", "continuous"])
    parser.add_argument("--capsule-workspace-source", default="all_levels",
                        choices=["final", "all_levels"])
    args = parser.parse_args()

    per_seed = {}
    for seed in SEEDS:
        cfg, tectum, *_ = _build_components(
            "dmts", action_dim=5, seed=seed, mock_semantic=False,
            load_tectum=args.load_tectum, latent_mode=args.latent_mode,
            capsule_workspace_source=args.capsule_workspace_source)
        per_seed[seed] = collect(cfg, tectum, args.episodes, seed)

    print("\n" + "=" * 84)
    print("FIDELITY CHECK (gates every number below)")
    print("=" * 84)
    print("  Recomputed kl_map.sum()/batch against the value the tectum squashed.\n")
    all_ok = True
    for seed, d in per_seed.items():
        ok, worst = fidelity(d)
        all_ok &= ok
        print(f"  seed {seed}  n={d['A'].shape[0]:>5}  worst relative error={worst:.3e}  "
              f"{'PASS' if ok else 'FAIL'}")
    if not all_ok:
        print("\n  FAILED. The recomputation does not reproduce sensory_tectum.py:443.")
        print("  This is a harness failure, NOT a finding about the agent. Stopping.")
        sys.exit(1)

    rows = {}
    for seed, d in per_seed.items():
        m = np.isin(d["phase"], PRIMARY)
        y2 = d["phase"][m]
        rng = np.random.default_rng(seed)
        y2_shuf = rng.permutation(y2)
        # Clock-free content test: sample_shape varies across trials and is independent
        # of step index, so this cannot be solved by reading elapsed time.
        s = d["phase"] == "sample"
        ys = d["shape"][s]
        ys_shuf = rng.permutation(ys)
        rows[seed] = {
            "A":  decode(d["A"][m],  y2, seed),
            "B1": decode(d["B1"][m], y2, seed),
            "B2": decode(d["B2"][m], y2, seed),
            "C":  decode(d["C"][m],  y2, seed),
            "D":  decode(d["B1"][m], y2_shuf, seed),
            "T":  decode(d["T"][m],  y2, seed),
            "B1_4c": decode(d["B1"], d["phase"], seed),
            "A_4c":  decode(d["A"],  d["phase"], seed),
            "C_4c":  decode(d["C"],  d["phase"], seed),
            "id_B1": decode(d["B1"][s], ys, seed),
            "id_C":  decode(d["C"][s],  ys, seed),
            "id_D":  decode(d["B1"][s], ys_shuf, seed),
            # TRIAL-grouped: the minimal fix for the identified leak. Gate reads these.
            "gA":  grouped_decode(d["A"][m],  y2, d["trial"][m], seed),
            "gB1": grouped_decode(d["B1"][m], y2, d["trial"][m], seed),
            "gB2": grouped_decode(d["B2"][m], y2, d["trial"][m], seed),
            "gC":  grouped_decode(d["C"][m],  y2, d["trial"][m], seed),
            "gD":  grouped_decode(d["B1"][m], y2_shuf, d["trial"][m], seed),
            "gT":  grouped_decode(d["T"][m],  y2, d["trial"][m], seed),
            "gid_B1": grouped_decode(d["B1"][s], ys, d["trial"][s], seed),
            "gid_C":  grouped_decode(d["C"][s],  ys, d["trial"][s], seed),
            "gid_D":  grouped_decode(d["B1"][s], ys_shuf, d["trial"][s], seed),
            # Shape is spatial: B2 and C2 keep the grid, B1 and C pool it away.
            "gid_B2": grouped_decode(d["B2"][s], ys, d["trial"][s], seed),
            "gid_C2": grouped_decode(d["C2"][s], ys, d["trial"][s], seed),
            "gid_D2": grouped_decode(d["B2"][s], ys_shuf, d["trial"][s], seed),
            # EPISODE-grouped: strictly harder, also demands cross-sequence transfer.
            "eA":  grouped_decode(d["A"][m],  y2, d["episode"][m], seed),
            "eB1": grouped_decode(d["B1"][m], y2, d["episode"][m], seed),
            "eB2": grouped_decode(d["B2"][m], y2, d["episode"][m], seed),
            "eC":  grouped_decode(d["C"][m],  y2, d["episode"][m], seed),
            "eD":  grouped_decode(d["B1"][m], y2_shuf, d["episode"][m], seed),
            "eT":  grouped_decode(d["T"][m],  y2, d["episode"][m], seed),
            "eid_B1": grouped_decode(d["B1"][s], ys, d["episode"][s], seed),
            "eid_C":  grouped_decode(d["C"][s],  ys, d["episode"][s], seed),
            "eid_D":  grouped_decode(d["B1"][s], ys_shuf, d["episode"][s], seed),
        }

    labels = {
        "A":  "A  kl_map.sum(), current bid input      dim    1",
        "B1": "B1 per-channel pooled kl_map            dim 1024",
        "B2": "B2 per-position pooled kl_map           dim  256",
        "C":  "C  obs_map, UPPER REFERENCE             dim   64",
        "D":  "D  B1 with SHUFFLED labels, NOISE FLOOR dim 1024",
        "T":  "T  step index in trial, CLOCK CONFOUND  dim    1",
    }

    print("\n" + "=" * 84)
    print(f"PRIMARY: {PRIMARY[0]} versus {PRIMARY[1]}, binary")
    print("=" * 84)
    print("  rnd  = random split over steps. LEAKS: near-duplicate neighbours both sides.")
    print("  TRI  = holds out whole trials. Minimal fix for that leak. GATE READS THIS.")
    print("  EPI  = holds out whole episodes. Also demands cross-sequence transfer.")
    for seed, r in rows.items():
        tmaj, emaj = r["gB1"]["majority"], r["eB1"]["majority"]
        print(f"\n  seed {seed}   n={r['B1']['n']}   TRI held-out={r['gB1']['n_test']} "
              f"(majority {tmaj:.4f})   EPI held-out={r['eB1']['n_test']} "
              f"(majority {emaj:.4f})")
        for k in ("A", "B1", "B2", "C", "D", "T"):
            rnd, tri, epi = (r[k]["test_acc"], r["g" + k]["test_acc"],
                             r["e" + k]["test_acc"])
            print(f"    {labels[k]}   rnd={rnd:.4f}  "
                  f"TRI={tri:.4f} ({tri - tmaj:+.4f})  EPI={epi:.4f} ({epi - emaj:+.4f})")

    print("\n" + "=" * 84)
    print("CLOCK-FREE CONTENT TEST: does kl_map carry stimulus identity?")
    print("=" * 84)
    print("  `sample_shape` during the sample phase. Varies across trials, independent")
    print("  of step index, so a clock cannot solve it.\n")
    print("  Shape is constant for all ~20 steps of a trial, so the random split is")
    print("  leaking here almost by construction. TRI is the number to read.\n")
    for seed, r in rows.items():
        tmaj, emaj = r["gid_B1"]["majority"], r["eid_B1"]["majority"]
        print(f"  seed {seed}  n={r['id_B1']['n']}  classes={r['id_B1']['n_classes']}  "
              f"TRI held-out={r['gid_B1']['n_test']} (majority {tmaj:.4f})  "
              f"EPI held-out={r['eid_B1']['n_test']} (majority {emaj:.4f})")
        for k, name in (("id_B1", "B1 kl_map  "), ("id_C", "C  obs_map "),
                        ("id_D", "D  SHUFFLED")):
            rnd, tri, epi = (r[k]["test_acc"], r["g" + k]["test_acc"],
                             r["e" + k]["test_acc"])
            print(f"    {name}  rnd={rnd:.4f}  TRI={tri:.4f} ({tri - tmaj:+.4f})  "
                  f"EPI={epi:.4f} ({epi - emaj:+.4f})")
        print("    -- space preserved (shape is a SPATIAL property) --")
        for k, name in (("gid_B2", "B2 kl_map  "), ("gid_C2", "C2 obs_map "),
                        ("gid_D2", "D2 SHUFFLED")):
            tri = r[k]["test_acc"]
            print(f"    {name}  TRI={tri:.4f} ({tri - tmaj:+.4f})")
        print(f"    held-out TRIALS ~ {r['gid_B1']['n_test'] // 20}  "
              f"(the effective sample for a 6-class identity test)")

    print("\n" + "=" * 84)
    print("SECONDARY: all four phases, reported not gated")
    print("=" * 84)
    for seed, r in rows.items():
        maj = r["B1_4c"]["majority"]
        print(f"  seed {seed}  majority={maj:.4f}  "
              f"A={r['A_4c']['test_acc']:.4f}  B1={r['B1_4c']['test_acc']:.4f}  "
              f"C={r['C_4c']['test_acc']:.4f}")

    print("\n" + "=" * 84)
    print("PRE-STATED GATE")
    print("=" * 84)
    print("  Read on the EPISODE-GROUPED split. The random split leaks.")
    verdicts = []
    for seed, r in rows.items():
        maj = r["gB1"]["majority"]
        b1, d_ = r["gB1"]["test_acc"], r["gD"]["test_acc"]
        if d_ - maj >= MARGIN:
            verdicts.append("INCONCLUSIVE")
        elif (b1 - d_) >= MARGIN and (b1 - maj) >= MARGIN:
            verdicts.append("REPAIRABLE")
        else:
            verdicts.append("NOT-REPAIRABLE")
    print(f"  per-seed verdicts: {verdicts}")

    agreed = set(verdicts)
    if len(agreed) != 1:
        print(f"\n  VERDICT: seeds DISAGREE {agreed}. Not decisive; report as such.")
    else:
        v = verdicts[0]
        print(f"\n  VERDICT: {v} at all {len(SEEDS)} seeds.")
        if v == "REPAIRABLE":
            print("  The map carries phase information the sum destroys. A pooled or")
            print("  weighted bid could produce task-tracking selection.")
        elif v == "NOT-REPAIRABLE":
            print("  No linear function of kl_map separates the phases. The sum is not")
            print("  what loses the information, so no bid formula recovers it.")
        else:
            print("  The shuffled control decodes above majority. The probe is")
            print("  overfitting and no number in this run is readable.")

    # The gate above is left exactly as pre-stated. The clock confound was identified
    # from the ENVIRONMENT SOURCE, not from these results, so it qualifies the reading
    # of the verdict rather than redefining the verdict after the fact.
    print("\n  CLOCK QUALIFIER on the verdict above:")
    for seed, r in rows.items():
        maj = r["gB1"]["majority"]
        t, idb1, idd = (r["gT"]["test_acc"] - maj,
                        r["gid_B1"]["test_acc"] - r["gid_B1"]["majority"],
                        r["gid_D"]["test_acc"] - r["gid_D"]["majority"])
        if t >= MARGIN and idb1 < MARGIN:
            note = ("a clock alone solves the phase test, and kl_map does NOT carry "
                    "stimulus identity. Read the phase decode as TIME, not content.")
        elif t >= MARGIN and idb1 >= MARGIN and (idb1 - idd) >= MARGIN:
            note = ("a clock alone solves the phase test, but kl_map ALSO carries "
                    "stimulus identity, which a clock cannot. Content is present.")
        elif t < MARGIN:
            note = "a clock does not solve the phase test; the confound is not active."
        else:
            note = "identity decode is at its own shuffle floor; content not shown."
        print(f"    seed {seed}: {note}")

    print("\n  Where the information dies, on the rule fixed in advance:")
    for seed, r in rows.items():
        maj = r["gB1"]["majority"]
        c, b1 = r["gC"]["test_acc"] - maj, r["gB1"]["test_acc"] - maj
        if c >= MARGIN and b1 < MARGIN:
            where = "reaches the tectum, discarded by the RSSM prediction error"
        elif c < MARGIN:
            where = "not present in obs_map either: points at the environment rendering"
        else:
            where = "present in both obs_map and kl_map"
        print(f"    seed {seed}: {where}")

    print("\n  LIMITATION: a linear probe failing shows no LINEAR function of kl_map")
    print("  separates the phases. It does not prove no function does.")


if __name__ == "__main__":
    main()
