# Hybrid Synthetic‑Video + Interactive Simulation Workflow (Open‑Source, 2025‑Q2)

> **Document status:** draft v0.4 – consolidated **12 May 2025**. Combines the latest open‑source T2V model landscape *and* the previously‑agreed hybrid training methodology, engineering deltas and cost‑benefit analysis.

---

## 1 Purpose

Provide a **licence‑clean**, scalable training pipeline that blends high‑throughput **text‑to‑video (T2V)** generation with **Unreal Engine 5 (UE5)** closed‑loop simulation to develop The Consciousness AI (ACM) without legal blockers for commercial deployment.

---

## 2 Landscape of Open‑Source T2V Models (2024‑25)

| Model (release)                               | Licence    | Clip / Res / FPS\*        | GPU RAM†             | Commercial | Notes                                 |
| --------------------------------------------- | ---------- | ------------------------- | -------------------- | ---------- | ------------------------------------- |
| **LTX‑Video 13B** (May 2025)                  | Apache‑2.0 | 6–8 s · 1216×704 · 30 FPS | 24 GB                | **Yes**    | Real‑time DiT; ComfyUI nodes ✔︎       |
| **Wan 2.1** (Feb 2025)                        | Apache‑2.0 | 8 s · 720 p · 24 FPS      | 20 GB                | **Yes**    | Multilingual prompts; tops VBench     |
| **MAGI‑1 24B / 4.5B** (Apr 2025)              | Apache‑2.0 | 12 s · 768 p · 24 FPS     | 32 GB / 8 GB         | **Yes**    | Autoregressive streaming; RTX 4090 OK |
| **Mochi 1** (Nov 2024)                        | Apache‑2.0 | 4 s · 480 p · 30 FPS      | 16 GB                | **Yes**    | High‑motion realism; SDXL‑compatible  |
| **Latte** (TMLR 2025)                         | Apache‑2.0 | 4 s · 768 p · 16 FPS      | 17 GB → 9 GB (8‑bit) | **Yes**    | Latent Diffusion Transformer          |
| **CogVideoX‑2B** (Aug 2024)                   | Apache‑2.0 | 5 s · 540 p · 16 FPS      | 12 GB                | **Yes**    | 5B variant non‑commercial             |
| **Open‑Sora 2.0** (Mar 2024)                  | Apache‑2.0 | 6 s · 720 p · 15 FPS      | 24 GB                | **Yes**    | Flow‑matching backbone                |
| **Pyramid Flow SD3** (Jan 2025)               | MIT        | 10 s · 768 p · 24 FPS     | 16 GB                | **Yes**    | Rectified Flow; FP8 inference         |
| **AnimateDiff** (ICLR 24)                     | Apache‑2.0 | 4 s · 512 p · 12 FPS      | 8 GB                 | **Yes**    | Motion LoRA over SDXL                 |
| **Latent Video Diffusion** (EleutherAI, 2024) | MIT        | 8 s · 512 p · 12 FPS      | 10 GB                | **Yes**    | Modular VAE; full training code       |

\* Default inference settings; most support longer clips via tiled sampling.
† Reported for bf16 unless noted.

---

## 3 Architecture Overview

```mermaid
graph TD
    subgraph Synthetic Generation (Offline)
        A1[LTXVideo]:::oss --> S
        A2[Wan2.1]:::oss --> S
        A3[MAGI‑1]:::oss --> S
        A4[Mochi1]:::oss --> S
        A5[Latte]:::oss --> S
        subgraph Optional
            A6[Open‑Sora]:::oss --> S
            A7[Pyramid Flow]:::oss --> S
            A8[AnimateDiff]:::oss --> S
        end
    end
    S[Clip Store] --> V(VideoDatasetEnv)
    V -->|latent dynamics pre‑train| O[Dreamer V3 Observation Model]

    subgraph Interactive Simulation (Online)
        O -->|weights| P[Policy + Value]
        P -->|actions| U[UE5 Env]
        U -->|observations| P
        U -->|expressive rigs| R[Emotion Reward Shaper]
    end
    classDef oss fill:#d2f4ea,stroke:#222;
```

CLI flag: `--generator=ltx|wan|magi|mochi|latte|opensora|pflow|animatediff` (multiple accepted).

---

## 4 Recommended Hybrid Workflow

### 4.1 Phase 1 – Synthetic‑Video Pre‑training

* **Generate** thousands of clips per scenario taxonomy (*survival, social, ethical, anomaly*).
* **Auto‑label** each clip with its text prompt, CLIP semantic tags and synthetic optical‑flow.
* **Pre‑train** modules offline:

  * **Vision encoders** (e.g., VideoLLaMA 3) on masked‑patch prediction.
  * **Emotional detector** using pseudo‑labels from facial‑affect VLM.
  * **Dreamer V3 observation model** on `(o_t, a=Ø, r=Ø, o_{t+1})` tuples.
* **Checkpoint** weights to `s3://acm-checkpoints/synth_pretrain/`.

### 4.2 Phase 2 – Transfer to UE5 (Online RL)

* **Load** pre‑trained weights into policy/value networks.
* **Interactive fine‑tuning** in UE5: full state‑action‑reward loop with physics‑grounded feedback.
* **Active‑learning**: log high‑uncertainty UE roll‑outs, convert to prompts, regenerate clips to close reality gap.

### 4.3 Phase 3 – Continual Learning

* Nightly cron checks upstream T2V releases (e.g., MAGI‑1 2.2), rebuilds affected prompts, re‑runs Phase 1 on delta set.
* SBOM diff ensures licence compatibility before merging into production dataset.

### 4.4 Engineering Changes

| Component                  | Action                                                                |
| -------------------------- | --------------------------------------------------------------------- |
| `video_dataset_env.py`     | New Gym‑style env exposing `step(done=True)` for offline trajectories |
| `simulation_controller.py` | Backend multiplexer (`video` vs `socket`)                             |
| `memory/pinecone`          | Store synthetic frames; set `physical_trust = 0.2`                    |
| `generators/*_runner.py`   | Wrappers for each backend listed in §2                                |
| CI pipeline                | SBOM build‑step + licence gate via `syft`                             |

---

## 5 Cost–Benefit Snapshot

| Aspect                 | Unreal Engine                               | T2V Synthetic Clips                    |
| ---------------------- | ------------------------------------------- | -------------------------------------- |
| **Interactivity**      | Full physics, multi‑agent                   | **None** – fixed trajectories          |
| **Data volume / cost** | GPU‑expensive but reusable                  | Cheap at scale (≈ \$0.18/clip @ 720 p) |
| **Physics fidelity**   | High                                        | Often wrong; no causality              |
| **Pipeline changes**   | Already integrated (`docs/architecture.md`) | Major rewiring for RL loop             |
| **Best use**           | Fine‑tuning, safety, conscious‑state eval   | Perception & world‑model pre‑training  |

---

## 6 Compliance Checklist

* **Apache‑2.0 / MIT** models: permissible for commercial use; ship licence headers with weights.
* **CogVideoX‑5B, HunyuanVideo**: research‑only flag; CI blocks export.
* **Stable Video Diffusion 1.1**: revenue cap ≤ US\$1 M; restrict to sandbox.

---

## 7 Risks & Mitigations

| Risk                        | Likelihood | Impact | Mitigation                                   |
| --------------------------- | ---------- | ------ | -------------------------------------------- |
| Model zoo bloat             | Med        | Med    | Monthly pruning; checkpoint dedup            |
| Licence regressions         | Med        | High   | SBOM diff + legal review gate                |
| Physics‑inconsistent priors | High       | Med    | Early switch to UE; decay synthetic features |
| Quality variance            | High       | Med    | Clip‑level FVD scoring; curriculum sampling  |

---

## 8 KPIs

* **≤ \$0.18/clip** at 720 p across backends.
* **≥ 25 % reduction** in UE steps to reach reward baseline vs v0.1.
* **Φ**\* consciousness score ≥ 0.7 after fine‑tuning.
* **0 licence violations** in production images.

---

## 9 Bottom Line

Text‑to‑video synthesis is a **powerful accelerator** for perception and offline world‑model learning, but it **cannot replace** the bidirectional, physics‑grounded, real‑time environment demanded by emergent‑consciousness architectures. Maintain Unreal Engine in the loop and treat T2V as a high‑throughput data generator—a complementary layer that slashes data costs without jeopardising embodiment, GNW ignition or safety testing.

---

## 10 References (selected 2024‑25)

1. Lightricks *LTX‑Video* repo — Apache‑2.0.
2. Alibaba *Wan 2.1* model card — Apache‑2.0.
3. Reuters, 25 Feb 2025 — Wan 2.1 open‑source release.
4. Sand AI *MAGI‑1* launch, Toolify (Apr 2025).
5. Reddit r/EnhancerAI — MAGI‑1 licence analysis (Apr 2025).
6. Genmo *Mochi 1* GitHub — Apache‑2.0.
7. VentureBeat, 30 Nov 2024 — Mochi 1 release article.
8. Latte GitHub — Apache‑2.0.
9. EleutherAI *Latent Video Diffusion* — MIT.
10. THUDM *CogVideoX* — Apache‑2.0 (2B) / restricted (5B).
11. Reddit r/StableDiffusion — licence caveats for CogVideoX‑5B.
12. Hugging Face blog, Feb 2025 — overview of open video‑gen models.
