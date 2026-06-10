"""Perception-decodability probe (Phase 5 prerequisite).

Question this answers: is the bottleneck perception or policy?

The 2026-06-02 localization compared reward-from-tap across pipeline stages, but
that comparison was confounded (epsilon schedule, MLP-vs-CNN learners, unequal
episode counts; see docs/results/agent_competence_fix_2026_06_02.md). This probe
removes those confounds by applying ONE linear decoder, with the same protocol, to
each representation stage and measuring how much task-relevant information survives.

Stages decoded (all from a no-grad replay of the real forward, train_rlhf.py:610-816):
  - pixels         : 32x32 block-mean of the raw frame (input sanity baseline)
  - obs_map        : tectum._last_obs_map (post retinotopic encoder + IE fusion)
  - tectum_content : tectum(frame, audio)[0] (post capsule collapse, 256-D)
  - broadcast      : settle-extracted tensor (post-GNW, 256-D; optional)

Labels (ground truth):
  - DMTS : sample shape / color / size at the SAMPLE phase (perception), and again
           at the DELAY phase (working-memory, sample off screen).
  - WCST : current card shape / color / count while the card is visible.

Decision rule (stated before running, FAILED-first):
  - pixel baseline near chance  -> the probe is broken; fix before interpreting.
  - obs_map / tectum_content decode well above chance and near pixels
                                -> perception preserves task info; bottleneck is
                                   downstream (policy/learning). Do NOT rebuild the
                                   front-end.
  - sharp drop pixels->obs_map  -> the encoder/fusion front-end loses task info;
                                   active-inference stage 1 is justified.
  - drop at obs_map->tectum_content -> the capsule collapse is the lossy stage.

Honesty caveats (also written into the results doc):
  - Components are probed UNTRAINED (static init). A low obs_map/tectum_content
    score under the conv fallback is partly the untrained 1x1 projection, not a
    fundamental architectural failure. The active encoder (DINOv2 vs conv fallback)
    is reported.
  - Single configuration, single seed. A measurement, not a law.

The training file is NOT modified; this script reuses its component builders.
"""
from __future__ import annotations

import argparse
import csv
import types
from pathlib import Path

import numpy as np
import torch

from scripts.training.train_rlhf import (
    build_config,
    init_components,
    frame_to_tensor,
    evaluate_emotion,
)
from simulations.environments.dmts_env import DMTSEnv
from simulations.environments.wcst_env import WCSTEnv


# --------------------------------------------------------------------------- #
# Linear decodability helper (importable + unit-tested)
# --------------------------------------------------------------------------- #
def linear_decode(X, y, seed: int = 0, test_size: float = 0.3):
    """Fit a linear probe X->y and return decodability on a held-out split.

    Same protocol for every stage so stages compare fairly: standardize features,
    fit a multinomial logistic regression (L2), score on a held-out test set.
    Falls back to a torch linear classifier if scikit-learn is unavailable.

    Returns a dict: test_acc, uniform_chance, majority, n, n_classes, method.
    test_acc is NaN when there are < 2 classes or < 10 samples.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    classes, y_idx = np.unique(y, return_inverse=True)
    n_classes = int(len(classes))
    n = int(len(y_idx))
    uniform_chance = 1.0 / n_classes if n_classes > 0 else float("nan")
    majority = float(np.bincount(y_idx).max() / n) if n > 0 else float("nan")

    if n_classes < 2 or n < 10:
        return {
            "test_acc": float("nan"), "uniform_chance": uniform_chance,
            "majority": majority, "n": n, "n_classes": n_classes, "method": "skip",
        }

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        from sklearn.model_selection import train_test_split

        stratify = y_idx if np.bincount(y_idx).min() >= 2 else None
        x_tr, x_te, y_tr, y_te = train_test_split(
            X, y_idx, test_size=test_size, random_state=seed, stratify=stratify
        )
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        clf.fit(x_tr, y_tr)
        acc = float(clf.score(x_te, y_te))
        method = "sklearn"
    except Exception:
        acc = _torch_probe(X, y_idx, n_classes, seed, test_size)
        method = "torch"

    return {
        "test_acc": acc, "uniform_chance": uniform_chance, "majority": majority,
        "n": n, "n_classes": n_classes, "method": method,
    }


def _torch_probe(X, y_idx, n_classes, seed, test_size):
    """Torch fallback: standardized linear classifier with a held-out split."""
    torch.manual_seed(seed)
    n = len(y_idx)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_test = max(1, int(n * test_size))
    te, tr = order[:n_test], order[n_test:]

    mu = X[tr].mean(axis=0)
    sd = X[tr].std(axis=0) + 1e-6
    xs = (X - mu) / sd

    x_tr = torch.tensor(xs[tr], dtype=torch.float32)
    y_tr = torch.tensor(y_idx[tr], dtype=torch.long)
    x_te = torch.tensor(xs[te], dtype=torch.float32)
    y_te = torch.tensor(y_idx[te], dtype=torch.long)

    lin = torch.nn.Linear(X.shape[1], n_classes)
    opt = torch.optim.Adam(lin.parameters(), lr=1e-2, weight_decay=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    for _ in range(300):
        opt.zero_grad()
        loss = loss_fn(lin(x_tr), y_tr)
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc = (lin(x_te).argmax(dim=1) == y_te).float().mean().item()
    return float(acc)


# --------------------------------------------------------------------------- #
# Component construction (reuses train_rlhf builders, no edits to that file)
# --------------------------------------------------------------------------- #
def _build_components(env_name: str, action_dim: int, seed: int, mock_semantic: bool,
                      load_tectum: str | None = None):
    args = types.SimpleNamespace(
        action_dim=action_dim, lr=1e-3, episodes=1, max_steps=200,
        env=env_name, enable_audio=False, enable_mock_semantic=mock_semantic,
        seed=seed,
    )
    config = build_config(args)
    config["device"] = "cpu"

    comps = init_components(config)
    # Positional unpack of the leading components this probe needs; trailing
    # training-only components (control-repr / reconstruction heads and optimizers)
    # are captured in *_ so future init_components growth does not break the probe.
    (tectum, workspace, reentrant, modulator, emotion_shaper, memory,
     action_core, semantic, gate, tectum_optimizer, reward_predictor,
     reward_optimizer, workspace_optimizer, auditory_specialist, self_model,
     rnd, rnd_optimizer, consolidation_mgr, riiu_phis, mock_sem,
     holonic_system, levin_evaluator, self_vector_module, self_vector_optimizer,
     *_) = comps

    # Forward-only probe: no phi needed. Detaching the gate keeps run_competition
    # from invoking pyphi every step (faster, and avoids the long-run segfault).
    if hasattr(workspace, "consciousness_gate"):
        workspace.consciousness_gate = None

    if load_tectum:
        state = torch.load(load_tectum, map_location="cpu")
        # h_state / z_state are RSSM runtime buffers saved mid-rollout, not learned
        # weights. Drop them: the probe resets tectum state per episode anyway.
        state = {k: v for k, v in state.items() if k not in ("h_state", "z_state")}
        missing, unexpected = tectum.load_state_dict(state, strict=False)
        leftover = [k for k in unexpected if k not in ("h_state", "z_state")]
        if missing or leftover:
            raise RuntimeError(
                f"tectum state_dict mismatch: missing={list(missing)} "
                f"unexpected={leftover}"
            )
        print(f"  loaded trained tectum from {load_tectum}")

    if isinstance(tectum, torch.nn.Module):
        tectum.eval()

    return config, tectum, workspace, reentrant, self_model, memory, mock_sem


def _downsample(obs: np.ndarray, k: int = 32) -> np.ndarray:
    """Block-mean downsample [H, W, 3] -> flat [k*k*3]."""
    h, w, c = obs.shape
    hs, ws = h // k, w // k
    cropped = obs[: hs * k, : ws * k, :]
    pooled = cropped.reshape(k, hs, k, ws, c).mean(axis=(1, 3))
    return pooled.reshape(-1).astype(np.float64)


def _compute_broadcast(config, tectum, workspace, reentrant, self_model,
                       memory, mock_sem, tectum_content, vision_bid, obs):
    """Replicate the run_episode forward (train_rlhf.py:646-816) to get broadcast.

    Returns a flat numpy vector, or None if anything fails (the core stage taps do
    not depend on this).
    """
    try:
        device = config["device"]
        ws_dim = config["workspace_dim"]
        emotion = evaluate_emotion(vision_bid, 0.0, 0.0)

        semantic_content = torch.zeros(1, ws_dim, device=device)
        semantic_bid = 0.0
        if mock_sem is not None and isinstance(obs, np.ndarray):
            emb = mock_sem.embed(obs)
            semantic_bid = mock_sem.bid_from_embedding(emb)
            if emb.shape[0] >= ws_dim:
                proj = emb[:ws_dim]
            else:
                proj = torch.nn.functional.pad(emb, (0, ws_dim - emb.shape[0]))
            semantic_content = proj.unsqueeze(0).to(device)

        raw_bids = {
            "vision": max(0.0, min(1.0, vision_bid)),
            "audio": 0.0,
            "memory": 0.1,
            "body": 0.05,
            "semantic": max(0.0, min(1.0, semantic_bid)),
        }
        vision_payload = {"tensor": tectum_content, "source": "tectum"}
        capsule_data = tectum.get_capsule_payload()
        if capsule_data:
            vision_payload.update(capsule_data)
        payloads = {
            "vision": vision_payload,
            "audio": {"tensor": torch.zeros(1, ws_dim, device=device), "source": "audio"},
            "semantic": {"tensor": semantic_content, "source": "semantic"},
        }
        intero = None
        if self_model is not None:
            intero = dict(self_model.state.interoceptive_state)

        result = reentrant.settle(
            workspace=workspace,
            specialists={"vision": tectum},
            initial_bids=raw_bids,
            payloads=payloads,
            goal_vector=torch.tensor([1.0, -1.0, 1.0], device=device),
            pad_state=emotion,
            interoceptive_state=intero,
        )
        bc = result.broadcast_content
        if isinstance(bc, torch.Tensor):
            t = bc
        elif isinstance(bc, dict) and isinstance(bc.get("_fused"), torch.Tensor):
            t = bc["_fused"]
        elif isinstance(bc, dict) and isinstance(bc.get("tensor"), torch.Tensor):
            t = bc["tensor"]
        else:
            return np.zeros(ws_dim, dtype=np.float64)
        if t.dim() == 1:
            t = t.unsqueeze(0)
        return t.detach().reshape(-1).cpu().numpy().astype(np.float64)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Collection
# --------------------------------------------------------------------------- #
def collect_dmts(episodes, seed, include_broadcast, mock_semantic, load_tectum=None):
    config, tectum, workspace, reentrant, self_model, memory, mock_sem = \
        _build_components("dmts", action_dim=5, seed=seed, mock_semantic=mock_semantic,
                          load_tectum=load_tectum)
    using_dino = bool(getattr(tectum.retinotopic_encoder, "using_dino", False))

    env = DMTSEnv(num_trials=20)
    records = []
    with torch.no_grad():
        for ep in range(episodes):
            obs, info = env.reset(seed=seed + ep)
            if hasattr(tectum, "reset_state"):
                tectum.reset_state(1)
            done = False
            steps = 0
            while not done and steps < 4000:
                phase = info.get("phase")
                frame = frame_to_tensor(obs, config["device"])
                audio = torch.zeros(1, config["tectum_feature_dim"], 2, device=config["device"])
                tectum_content, vision_bid = tectum(frame, audio)

                if phase in ("sample", "delay"):
                    rec = {
                        "group": "sample" if phase == "sample" else "delay",
                        "pixels": _downsample(obs),
                        "obs_map": tectum._last_obs_map.reshape(-1).cpu().numpy().astype(np.float64),
                        "tectum_content": tectum_content.reshape(-1).cpu().numpy().astype(np.float64),
                        "labels": {
                            "shape": info["sample_shape"],
                            "color": info["sample_color"],
                            "size": info["sample_size"],
                        },
                    }
                    if include_broadcast:
                        bc = _compute_broadcast(config, tectum, workspace, reentrant,
                                                self_model, memory, mock_sem,
                                                tectum_content, vision_bid, obs)
                        if bc is not None:
                            rec["broadcast"] = bc
                    records.append(rec)

                action = 0 if phase != "choice" else int(np.random.randint(1, env.num_choices + 1))
                obs, _, term, trunc, info = env.step(action)
                done = term or trunc
                steps += 1
    return records, using_dino


def collect_wcst(episodes, seed, include_broadcast, mock_semantic, load_tectum=None):
    config, tectum, workspace, reentrant, self_model, memory, mock_sem = \
        _build_components("wcst", action_dim=4, seed=seed, mock_semantic=mock_semantic,
                          load_tectum=load_tectum)
    using_dino = bool(getattr(tectum.retinotopic_encoder, "using_dino", False))

    env = WCSTEnv(num_trials=200)
    records = []
    rng = np.random.default_rng(seed)
    with torch.no_grad():
        for ep in range(episodes):
            obs, info = env.reset(seed=seed + ep)
            if hasattr(tectum, "reset_state"):
                tectum.reset_state(1)
            done = False
            steps = 0
            while not done and steps < 4000:
                card = getattr(env, "_current_card", None)
                in_feedback = getattr(env, "_feedback_remaining", 0) > 0
                frame = frame_to_tensor(obs, config["device"])
                audio = torch.zeros(1, config["tectum_feature_dim"], 2, device=config["device"])
                tectum_content, vision_bid = tectum(frame, audio)

                if card is not None and not in_feedback:
                    rec = {
                        "group": "card",
                        "pixels": _downsample(obs),
                        "obs_map": tectum._last_obs_map.reshape(-1).cpu().numpy().astype(np.float64),
                        "tectum_content": tectum_content.reshape(-1).cpu().numpy().astype(np.float64),
                        "labels": {
                            "shape": card["shape"],
                            "color": card["color"],
                            "count": int(card["count"]),
                        },
                    }
                    if include_broadcast:
                        bc = _compute_broadcast(config, tectum, workspace, reentrant,
                                                self_model, memory, mock_sem,
                                                tectum_content, vision_bid, obs)
                        if bc is not None:
                            rec["broadcast"] = bc
                    records.append(rec)

                action = int(rng.integers(0, 4))
                obs, _, term, trunc, info = env.step(action)
                done = term or trunc
                steps += 1
    return records, using_dino


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
_STAGES = ["pixels", "obs_map", "tectum_content", "broadcast"]


def _evaluate(records, group, label_keys, seed):
    """Return rows: (group, label, stage, result-dict) for one record group."""
    rows = []
    subset = [r for r in records if r["group"] == group]
    if not subset:
        return rows
    for label in label_keys:
        y = [r["labels"][label] for r in subset]
        for stage in _STAGES:
            if stage not in subset[0]:
                continue
            X = [r[stage] for r in subset if stage in r]
            y_stage = [r["labels"][label] for r in subset if stage in r]
            res = linear_decode(X, y_stage, seed=seed)
            rows.append((group, label, stage, res))
    return rows


def _print_and_collect(env_name, rows, using_dino, all_csv):
    enc = "DINOv2" if using_dino else "conv-fallback (UNTRAINED projection)"
    print(f"\n=== {env_name.upper()}  (visual encoder: {enc}) ===")
    print(f"{'group':8} {'label':7} {'stage':16} {'test_acc':>9} "
          f"{'chance':>7} {'major':>7} {'n':>6} {'cls':>4}")
    for group, label, stage, res in rows:
        acc = res["test_acc"]
        acc_s = "  nan  " if np.isnan(acc) else f"{acc:7.3f}"
        print(f"{group:8} {label:7} {stage:16} {acc_s:>9} "
              f"{res['uniform_chance']:7.3f} {res['majority']:7.3f} "
              f"{res['n']:6d} {res['n_classes']:4d}")
        all_csv.append({
            "env": env_name, "encoder": enc, "group": group, "label": label,
            "stage": stage, "test_acc": acc, "uniform_chance": res["uniform_chance"],
            "majority": res["majority"], "n": res["n"], "n_classes": res["n_classes"],
            "method": res["method"],
        })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=8,
                        help="Episodes per environment for collection")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-broadcast", action="store_true",
                        help="Skip the post-GNW broadcast tap (faster smoke runs)")
    parser.add_argument("--mock-semantic", action="store_true",
                        help="Enable the deterministic mock-semantic module")
    parser.add_argument("--envs", type=str, default="dmts,wcst",
                        help="Comma-separated subset of {dmts,wcst}")
    parser.add_argument("--out-dir", type=str, default="runs/perception_probe")
    parser.add_argument("--load-tectum", type=str, default=None,
                        help="Load a trained tectum state_dict (from train_rlhf "
                             "--save-tectum) before probing, to measure trained "
                             "perception instead of the untrained init.")
    args = parser.parse_args()

    include_broadcast = not args.no_broadcast
    envs = [e.strip() for e in args.envs.split(",") if e.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_csv = []

    if "dmts" in envs:
        print("Collecting DMTS ...")
        recs, dino = collect_dmts(args.episodes, args.seed, include_broadcast,
                                  args.mock_semantic, args.load_tectum)
        print(f"  collected {len(recs)} labeled frames")
        rows = _evaluate(recs, "sample", ["shape", "color", "size"], args.seed)
        rows += _evaluate(recs, "delay", ["shape", "color", "size"], args.seed)
        _print_and_collect("dmts", rows, dino, all_csv)

    if "wcst" in envs:
        print("Collecting WCST ...")
        recs, dino = collect_wcst(args.episodes, args.seed, include_broadcast,
                                  args.mock_semantic, args.load_tectum)
        print(f"  collected {len(recs)} labeled frames")
        rows = _evaluate(recs, "card", ["shape", "color", "count"], args.seed)
        _print_and_collect("wcst", rows, dino, all_csv)

    csv_path = out_dir / "decodability.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "env", "encoder", "group", "label", "stage", "test_acc",
            "uniform_chance", "majority", "n", "n_classes", "method",
        ])
        writer.writeheader()
        for row in all_csv:
            writer.writerow(row)
    print(f"\nWrote {csv_path}")


if __name__ == "__main__":
    main()
