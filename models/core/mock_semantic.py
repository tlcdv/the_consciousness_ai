"""Deterministic semantic module for Phi-1 testability.

Phase D of the 2026-05-17 Phi-1 retest plan
(~/.claude/plans/let-s-plan-the-next-misty-parasol.md).

The semantic pathway (`models/core/semantic_pathway.py`) requires Qwen2-VL
to produce non-zero embeddings. Without Qwen2-VL loaded, the semantic
channel bids 0 and does not participate in workspace competition. In the
2026-05-17 substrate probe this left dark_room with only one active
module (vision), which made AKOrN binding non-trivial and Phi-1 physically
untestable.

`MockSemanticModule` produces a deterministic 1536-D embedding derived from
the observation frame (hashed-bin projection of pixel statistics). It is
explicit about being a mock: no learning, no claim to capture real semantics,
just a deterministic non-zero signal that varies with the input so that
semantic-bid sync_R contribution is non-zero.

When `--enable-mock-semantic` is on AND Qwen2-VL is not loaded, this module
replaces the zero stub. Its bid is the L2 norm of the embedding, clipped
to [0, 1]; bid >= 0.1 in typical use.
"""
from __future__ import annotations

import numpy as np
import torch


class MockSemanticModule:
    """Deterministic semantic embedding producer for Phi-1 testability.

    Constructor:
        embedding_dim: output dimensionality (default 1536 to match Qwen2-VL)
        hash_bins: number of hash bins for the projection (default 64)
        seed: deterministic seed for the projection weights (default 42)
    """

    def __init__(self, embedding_dim: int = 1536, hash_bins: int = 64, seed: int = 42):
        self.embedding_dim = embedding_dim
        self.hash_bins = hash_bins
        # Deterministic projection from hash_bins to embedding_dim. Frozen.
        # Reproducible across runs (same seed -> same projection).
        rng = np.random.default_rng(seed)
        self._projection = torch.tensor(
            rng.normal(0.0, 1.0 / np.sqrt(hash_bins), size=(hash_bins, embedding_dim)),
            dtype=torch.float32,
        )

    def _hash_features(self, obs: np.ndarray) -> np.ndarray:
        """Extract hash_bins features from the observation.

        Strategy: tile the obs into roughly sqrt(hash_bins/2) x sqrt(hash_bins/2)
        regions, compute (mean, std) per region. Standard deviation is much
        more discriminative than mean alone on realistic frames where the
        agent/light occupies a small region; mean alone produces near-uniform
        features on random or low-contrast obs.
        """
        # Reduce to grayscale by mean over channels
        if obs.ndim == 3:
            gray = obs.mean(axis=-1).astype(np.float32) / 255.0
        else:
            gray = obs.astype(np.float32) / 255.0
        # Tile into sqrt(hash_bins/2) x sqrt(hash_bins/2). Each tile gives
        # 2 features (mean, std), so total = side*side*2 ~= hash_bins.
        side = max(1, int(np.sqrt(self.hash_bins // 2)))
        h, w = gray.shape
        tile_h = max(1, h // side)
        tile_w = max(1, w // side)
        features = np.zeros(self.hash_bins, dtype=np.float32)
        idx = 0
        for i in range(side):
            for j in range(side):
                if idx + 1 >= self.hash_bins:
                    break
                tile = gray[i * tile_h:(i + 1) * tile_h, j * tile_w:(j + 1) * tile_w]
                features[idx] = tile.mean()
                features[idx + 1] = tile.std()
                idx += 2
        return features

    def embed(self, obs: np.ndarray) -> torch.Tensor:
        """Produce a 1-D embedding tensor of shape [embedding_dim] from an obs.

        Output is deterministic given the same obs and seed; the projection
        is frozen at construction.
        """
        features = torch.tensor(self._hash_features(obs), dtype=torch.float32)
        embedding = features @ self._projection
        # Normalize to give a stable bid magnitude
        norm = embedding.norm() + 1e-6
        return embedding / norm * float(np.tanh(norm.item()))

    def bid_from_embedding(self, embedding: torch.Tensor) -> float:
        """Compute a scalar bid from an embedding.

        The bid is the embedding's L2 norm clipped to [0.1, 1.0]. Floor at 0.1
        so this module always participates in workspace competition with a
        non-trivial bid (the whole point is to be a non-zero alternative to
        the zero stub).
        """
        return float(np.clip(embedding.norm().item(), 0.1, 1.0))
