from __future__ import annotations

import torch
import numpy as np
import time
import logging
from dataclasses import dataclass

try:
    import faiss
except ImportError:
    faiss = None
    logging.warning("FAISS not installed. EmotionalMemoryIndex will use brute force search.")

try:
    from models.emotion.tgnn.emotional_graph import EmotionalGraphNetwork
except ImportError:
    EmotionalGraphNetwork = None

try:
    from models.evaluation.consciousness_metrics import ConsciousnessMetrics
except ImportError:
    ConsciousnessMetrics = None


@dataclass
class MemoryIndexConfig:
    """Configuration for emotional memory indexing."""
    vector_dimension: int = 768
    index_name: str = "emotional-memories"
    metric: str = "cosine"
    embedding_batch_size: int = 32


class EmotionalMemoryIndex:
    """
    Indexes and retrieves emotional memories using vector similarity.

    Uses FAISS for fast local vector search instead of an external service.
    Falls back to brute force numpy search if FAISS is unavailable.

    Features:
    1. Emotional context embedding
    2. Fast similarity search via FAISS
    3. Temporal coherence tracking
    4. Consciousness relevant retrieval
    """

    def __init__(self, config):
        """
        Initialize the emotional memory index.

        Args:
            config: MemoryIndexConfig or dict with index parameters.
        """
        if isinstance(config, dict):
            self.config = MemoryIndexConfig(
                vector_dimension=config.get('vector_dimension', 768),
                index_name=config.get('index_name', 'emotional-memories'),
                metric=config.get('metric', 'cosine'),
                embedding_batch_size=config.get('embedding_batch_size', 32),
            )
        else:
            self.config = config

        self.emotion_network = None
        self.consciousness_metrics = None

        # In memory storage
        self._vectors: list[np.ndarray] = []
        self._metadata: list[dict] = []
        self._faiss_index = None
        self.total_memories = 0

        self.memory_stats = {
            "emotional_coherence": 0.0,
            "temporal_consistency": 0.0,
            "consciousness_relevance": 0.0,
        }

        self._init_vector_store()

    def _init_vector_store(self) -> None:
        """Initialize the FAISS index or prepare brute force fallback."""
        dim = self.config.vector_dimension
        if faiss is not None:
            # Use inner product index. For cosine similarity we normalize vectors before insert.
            self._faiss_index = faiss.IndexFlatIP(dim)
            logging.info(f"FAISS index initialized with dimension {dim}.")
        else:
            logging.info("Using brute force numpy search (FAISS unavailable).")

    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        """L2 normalize a vector for cosine similarity via inner product."""
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec
        return vec / norm

    def store_memory(
        self,
        state: torch.Tensor,
        emotion_values: dict[str, float],
        attention_level: float,
        narrative: str,
        context: dict | None = None,
    ) -> str:
        """
        Store an emotional memory with indexed metadata.

        Args:
            state: Tensor representing state or environment info.
            emotion_values: dict of emotional signals (valence, arousal, dominance).
            attention_level: Numeric indicator of attention/consciousness.
            narrative: Text describing the experience.
            context: Optional dict for extra metadata (timestamps, etc).

        Returns:
            A string memory ID.
        """
        # Generate emotional embedding
        if self.emotion_network is not None:
            emotional_embedding = self.emotion_network.get_embedding(emotion_values)
            vector = emotional_embedding.detach().cpu().numpy().flatten()
        else:
            # Fallback: build vector from emotion values only (matches retrieval query construction)
            vector = np.array([emotion_values.get('valence', 0.0),
                               emotion_values.get('arousal', 0.0),
                               emotion_values.get('dominance', 0.0)], dtype=np.float32)

        # Calculate consciousness relevance
        if self.consciousness_metrics is not None:
            awareness_result = self.consciousness_metrics.evaluate_emotional_awareness([
                {
                    "state": state,
                    "emotion": emotion_values,
                    "attention": attention_level,
                    "narrative": narrative,
                }
            ])
            consciousness_score = awareness_result.get("mean_emotional_awareness", 0.0)
        else:
            # Fallback: derive from attention level and emotion intensity
            emo_intensity = sum(abs(v) for v in emotion_values.values()) / max(len(emotion_values), 1)
            consciousness_score = (attention_level + emo_intensity) / 2.0

        # Pad or truncate to match configured dimension
        dim = self.config.vector_dimension
        if vector.shape[0] < dim:
            vector = np.pad(vector, (0, dim - vector.shape[0]))
        elif vector.shape[0] > dim:
            vector = vector[:dim]

        vector = self._normalize(vector).astype(np.float32)

        memory_id = f"memory_{self.total_memories}"
        timestamp = 0.0
        if context and "timestamp" in context:
            timestamp = context["timestamp"]
        else:
            timestamp = time.time()

        metadata = {
            "id": memory_id,
            "emotion_values": emotion_values,
            "attention_level": float(attention_level),
            "narrative": narrative,
            "consciousness_score": float(consciousness_score),
            "timestamp": timestamp,
        }

        # Store
        self._vectors.append(vector)
        self._metadata.append(metadata)

        if self._faiss_index is not None:
            self._faiss_index.add(vector.reshape(1, -1))

        self.total_memories += 1
        self._update_memory_stats(consciousness_score)
        return memory_id

    def retrieve_similar_memories(
        self,
        emotion_query: dict[str, float],
        k: int = 5,
        min_consciousness_score: float = 0.5,
    ) -> list[dict]:
        """
        Retrieve similar memories based on emotional context.

        Args:
            emotion_query: dict of emotion signals to build the query vector.
            k: Number of results to return after filtering.
            min_consciousness_score: Minimum consciousness score to include.

        Returns:
            list of memory dicts with id, emotion_values, attention_level,
            narrative, consciousness_score, and similarity.
        """
        if self.total_memories == 0:
            return []

        if self.emotion_network is not None:
            query_embedding = self.emotion_network.get_embedding(emotion_query)
            query_vec = query_embedding.detach().cpu().numpy().flatten()
        else:
            query_vec = np.array([emotion_query.get('valence', 0.0),
                                  emotion_query.get('arousal', 0.0),
                                  emotion_query.get('dominance', 0.0)], dtype=np.float32)

        dim = self.config.vector_dimension
        if query_vec.shape[0] < dim:
            query_vec = np.pad(query_vec, (0, dim - query_vec.shape[0]))
        elif query_vec.shape[0] > dim:
            query_vec = query_vec[:dim]

        query_vec = self._normalize(query_vec).astype(np.float32)

        # Fetch more than needed so we can filter
        fetch_k = min(k * 3, self.total_memories)

        if self._faiss_index is not None and self._faiss_index.ntotal > 0:
            scores, indices = self._faiss_index.search(query_vec.reshape(1, -1), fetch_k)
            scores = scores[0]
            indices = indices[0]
        else:
            # Brute force cosine similarity
            all_vecs = np.array(self._vectors)
            scores = all_vecs @ query_vec
            indices = np.argsort(-scores)[:fetch_k]
            scores = scores[indices]

        memories = []
        for score, idx in zip(scores, indices):
            if idx < 0 or idx >= len(self._metadata):
                continue
            meta = self._metadata[idx]
            if meta["consciousness_score"] >= min_consciousness_score:
                memories.append({
                    "id": meta["id"],
                    "emotion_values": meta["emotion_values"],
                    "attention_level": meta["attention_level"],
                    "narrative": meta["narrative"],
                    "consciousness_score": meta["consciousness_score"],
                    "similarity": float(score),
                })

        memories.sort(
            key=lambda x: (x["similarity"] + x["consciousness_score"]) / 2.0,
            reverse=True,
        )
        return memories[:k]

    def get_temporal_sequence(
        self,
        start_time: float,
        end_time: float,
        min_consciousness_score: float = 0.0,
    ) -> list[dict]:
        """
        Retrieve memories within a given time window.

        Args:
            start_time: Start of time window.
            end_time: End of time window.
            min_consciousness_score: Filter out memories below this threshold.

        Returns:
            list of memory dicts sorted by timestamp.
        """
        memories = []
        for meta in self._metadata:
            ts = meta.get("timestamp", 0.0)
            if start_time <= ts <= end_time and meta["consciousness_score"] >= min_consciousness_score:
                memories.append({
                    "id": meta["id"],
                    "emotion_values": meta["emotion_values"],
                    "attention_level": meta["attention_level"],
                    "narrative": meta["narrative"],
                    "consciousness_score": meta["consciousness_score"],
                    "timestamp": ts,
                })
        memories.sort(key=lambda x: x["timestamp"])
        return memories

    def _update_memory_stats(self, consciousness_score: float) -> None:
        """Update memory stats incrementally."""
        alpha = 0.01
        old_val = self.memory_stats["consciousness_relevance"]
        self.memory_stats["consciousness_relevance"] = (1 - alpha) * old_val + alpha * consciousness_score

        # Update coherence based on recent memories
        if len(self._metadata) >= 2:
            last_two = self._metadata[-2:]
            self.memory_stats["temporal_consistency"] = self._calculate_temporal_consistency(
                last_two[0], last_two[1]
            )
            recent_attention = [m["attention_level"] for m in self._metadata[-10:]]
            self.memory_stats["emotional_coherence"] = float(np.mean(recent_attention))

    def _calculate_temporal_consistency(self, m1: dict, m2: dict) -> float:
        """Compare two memories to produce a consistency measure from 0.0 to 1.0."""
        emo_diff = []
        for k in m1["emotion_values"]:
            if k in m2["emotion_values"]:
                emo_diff.append(abs(m1["emotion_values"][k] - m2["emotion_values"][k]))
        emotion_consistency = 1.0 - (np.mean(emo_diff) if emo_diff else 0.0)

        cs_diff = abs(m1["consciousness_score"] - m2["consciousness_score"])
        consciousness_consistency = 1.0 - cs_diff

        return float((emotion_consistency + consciousness_consistency) / 2.0)
