"""
Is the oscillatory binding doing anything, and does a per-episode phase reset change it?

Oscillatory binding is one of the six neurobiological features that guide this
architecture (`docs/architecture.md`). Its stated role: "Kuramoto oscillators
synchronize module representations. Modules that process related information
phase-lock into unified percepts" (`architecture.md:23`). The binding boost at
`oscillatory_binding.py:193-199` is described in the code as "an emergent property of
the oscillator dynamics".

Two facts make that worth checking directly:

  1. `reset_state` is defined at `oscillatory_binding.py:149` and is NEVER called from
     `scripts/training/train_rlhf.py`. The only `reset_state` there is
     `tectum.reset_state(1)` at `:423`, a different object.
  2. The phases are initialized ONCE per process, lazily, only when
     `current_phases is None` (`oscillatory_binding.py:170-171`), and are then carried
     forward across every step of every episode.

So the oscillators are never re-randomized. `sync_r_content_2026_09.md` already
established that they converge to full synchrony and that sync_R then equals the mean
of the bids. This probe asks the next question: once converged, does the binding layer
still DISCRIMINATE between modules, and does resetting restore discrimination.

ARMS
    A   current behaviour, no reset
    B   `binding_system.reset_state()` at the start of every episode

## CONTENT TEST (--content), and its PRE-STATED GATE

Unfreezing the layer is not the same as making it bind. The claim under test is
`architecture.md:23`, "modules that process related information phase-lock". The
testable version on DMTS, where only vision carries real content, is whether the
5-module ALIGNMENT VECTOR carries the stimulus.

    (a) BINDING CARRIES CONTENT
        the alignment vector decodes 6-class `sample_shape` above its own permutation
        null p95, in arm B.
    (b) FROZEN
        fewer than 3 distinct alignment vectors. Expected in arm A, which is the
        control: a frozen layer cannot decode anything and must read at chance.
    (c) RELAXATION NOISE
        arm B varies and does NOT clear its null. The reset produces a settling
        transient and nothing more.

The effective sample is TRIALS, not steps, because `sample_shape` is constant within a
trial. Alignment is averaged per trial and one trial is one reading, so there is no
pseudo-replication. Permutation shuffles the shape label across trials.

A time confound cannot manufacture a result here: alignment after a reset is a
function of steps-since-reset, but `DMTSEnv._start_new_trial` draws `sample_shape`
at random per trial, so shape is independent of position in the episode.

MEASURED, per step, after a 400-step warm-up

    sync_R          the Kuramoto order parameter
    align spread    max minus min of the per-module alignment, recomputed exactly as
                    `bind_bids` computes it. The boost is `1 + 0.5 * align`, so if the
                    spread is constant the boost is a fixed multiplier applied to every
                    module and the binding cannot change anyone's rank. This is the
                    quantity that decides whether binding is selective.
    winner          which module the workspace selects

A NOTE ON REPLICATION, because the output invites a wrong reading. All three trained
checkpoints return IDENTICAL numbers. That is expected and is not three-seed
replication. The oscillator's only input is the bid vector. The vision bid is
`tanh(kl_div)`, saturated at exactly 1.0 on every checkpoint, and the other four bids
are constants, so the oscillator receives the same input regardless of the trained
weights. Treat this as ONE measurement, plus a demonstration that the saturated bid
makes the binding layer blind to training.

Read-only. No training, no checkpoint written, no model modified.

Run:
    python -m scripts.analysis.probe_oscillator_reset --checkpoint runs/gate3_s42
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.analysis.probe_bid_counterfactual import OPPONENTS, _make_chain  # noqa: E402
from scripts.analysis.probe_perception_decodability import (  # noqa: E402
    _build_components,
    evaluate_emotion,
)
from scripts.training.train_rlhf import frame_to_tensor  # noqa: E402
from simulations.environments.dmts_env import DMTSEnv  # noqa: E402


def alignment_of(binding_system):
    """Per-module alignment to the mean field, exactly as bind_bids computes it."""
    ph = binding_system.current_phases
    if ph is None:
        return None
    mean_field = torch.mean(ph, dim=1, keepdim=True)
    a = F.cosine_similarity(ph, mean_field.expand_as(ph), dim=-1)
    return ((a + 1.0) / 2.0)[0].tolist()


def run(args, reset_each_episode: bool):
    config, tectum, *_ = _build_components(
        "dmts", action_dim=5, seed=args.seed, mock_semantic=False,
        load_tectum=os.path.join(args.checkpoint, "tectum.pt"),
        latent_mode=args.latent_mode,
        capsule_workspace_source=args.capsule_workspace_source)
    tectum.eval()
    ws, re = _make_chain(config, args.replay_seed)
    device = config["device"]
    goal = torch.tensor([1.0, -1.0, 1.0], device=device)
    zero = torch.zeros(1, config["workspace_dim"], device=device)
    env = DMTSEnv(num_trials=20)
    syncs, spreads, winners = [], [], []
    aligns, shapes, trials = [], [], []
    with torch.no_grad():
        for ep in range(args.episodes):
            obs, info = env.reset(seed=args.seed + ep)
            tectum.reset_state(1)
            if reset_each_episode:
                ws.binding_system.reset_state()
            done, steps = False, 0
            while not done and steps < args.max_steps:
                frame = frame_to_tensor(obs, device)
                audio = torch.zeros(1, config["tectum_feature_dim"], 2, device=device)
                content, bid = tectum(frame, audio)
                pl = {"vision": {"tensor": content, "source": "tectum"},
                      "audio": {"tensor": zero, "source": "audio"},
                      "semantic": {"tensor": zero, "source": "semantic"}}
                cap = tectum.get_capsule_payload()
                if cap:
                    pl["vision"].update(cap)
                bids = dict(OPPONENTS["bcast"])
                bids["vision"] = float(max(0.0, min(1.0, bid)))
                re.settle(workspace=ws, specialists={"vision": tectum},
                          initial_bids=bids, payloads=pl, goal_vector=goal,
                          pad_state=evaluate_emotion(bids["vision"], 0.0, 0.0),
                          interoceptive_state=None)
                syncs.append(float(getattr(ws, "last_sync_R", 0.0)))
                al = alignment_of(ws.binding_system)
                spreads.append(float(max(al) - min(al)) if al else float("nan"))
                aligns.append(list(al) if al else [float("nan")] * 5)
                shapes.append(info.get("sample_shape"))
                trials.append("%d:%s" % (ep, info.get("trial")))
                w = getattr(ws.state, "winners", []) or []
                winners.append(w[0] if w else "")
                obs, _, t, tr, info = env.step(0)
                done = t or tr
                steps += 1
    w0 = args.warmup
    return {
        "sync": np.array(syncs)[w0:], "spread": np.array(spreads)[w0:],
        "winner": np.array(winners)[w0:], "align": np.array(aligns)[w0:],
        "shape": np.array(shapes)[w0:], "trial": np.array(trials)[w0:],
    }


def content_test(d, n_perm: int, seed: int):
    """Does the 5-module alignment vector decode 6-class sample_shape?

    One reading per TRIAL, because sample_shape is constant within a trial.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    al, sh, tr = d["align"], d["shape"], d["trial"]
    X, y = [], []
    for t in dict.fromkeys(tr.tolist()):
        m = tr == t
        if m.sum() < 5:
            continue
        X.append(al[m].mean(axis=0))
        y.append(sh[m][0])
    X = np.asarray(X, dtype=np.float64)
    cls = sorted(set(y))
    y = np.asarray([cls.index(v) for v in y])
    counts = np.bincount(y)
    if len(X) < 20 or counts.min() < 2:
        print("     too few trials (%d) or classes for a content test" % len(X))
        return
    keep = X.std(axis=0) > 0
    if int(keep.sum()) == 0:
        print("     alignment is CONSTANT across trials: nothing to decode  (gate b)")
        return
    X = X[:, keep]
    n_splits = max(2, min(5, int(counts.min())))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    acc = float(cross_val_score(clf, X, y, cv=cv).mean())
    rng = np.random.default_rng(seed)
    null = np.array([float(cross_val_score(clf, X, rng.permutation(y), cv=cv).mean())
                     for _ in range(n_perm)])
    p95 = float(np.percentile(null, 95))
    print("     trials=%d  live dims=%d of 5  %d-fold CV" % (len(y), int(keep.sum()), n_splits))
    print("     acc=%.4f   null p95=%.4f   null mean=%.4f   uniform=%.4f  majority=%.4f"
          % (acc, p95, float(null.mean()), 1.0 / len(cls), float(counts.max() / len(y))))
    print("     VERDICT: %s" % ("CARRIES CONTENT (a)" if acc > p95
                                else "RELAXATION NOISE (c)"))



def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default="runs/gate3_s42")
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--warmup", type=int, default=400)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--replay-seed", type=int, default=20260913)
    p.add_argument("--content", action="store_true",
                   help="Run the content test. See the pre-stated gate.")
    p.add_argument("--permutations", type=int, default=500)
    p.add_argument("--latent-mode", default="continuous",
                   choices=["discrete", "continuous"])
    p.add_argument("--capsule-workspace-source", default="all_levels",
                   choices=["final", "all_levels"])
    args = p.parse_args()

    print("Checkpoint: %s   %d episodes, %d warm-up steps discarded\n"
          % (args.checkpoint, args.episodes, args.warmup))
    for label, reset in (("A  no reset (current behaviour)", False),
                         ("B  reset_state() per episode", True)):
        d = run(args, reset)
        sy, sp, w = d["sync"], d["spread"], d["winner"]
        live = w[w != ""]
        print("%s   n=%d" % (label, sy.size))
        print("   sync_R        distinct=%-6d min=%.6f max=%.6f  sd=%.3e"
              % (len(np.unique(np.round(sy, 9))), sy.min(), sy.max(), sy.std()))
        print("   align spread  distinct=%-6d min=%.3e max=%.3e  mean=%.3e"
              % (len(np.unique(np.round(sp, 9))), sp.min(), sp.max(), sp.mean()))
        if live.size:
            vals, counts = np.unique(live, return_counts=True)
            top = sorted(zip(vals, counts), key=lambda x: -x[1])[:3]
            shares = ", ".join("%s=%.3f" % (v, c / live.size) for v, c in top)
        else:
            shares = "(none)"
        print("   winner        %s   silent=%.3f"
              % (shares, float((w == "").mean())))
        if args.content:
            print("   CONTENT TEST, 6-class sample_shape, one reading per trial")
            content_test(d, args.permutations, args.replay_seed)
        print()
    print("A constant `align spread` means the binding boost is the same multiplier for")
    print("every module, so the binding cannot change which module wins.")


if __name__ == "__main__":
    main()
