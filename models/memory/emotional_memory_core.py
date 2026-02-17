"""
Emotional memory system implementing:
- Storage for RL experiences (state, action, reward, emotion, next_state).
- Persistence across simulations (loading/saving).
- Contextual retrieval using placeholders for vector similarity and emotional relevance.
- Support for associating experiences with model adaptations (e.g., LoRAs).
- Meta-memory concepts (placeholder).
"""

import logging
import time
import os
import pickle # Using pickle for simplicity, consider safer alternatives (JSON lines, DB) for production
from typing import Dict, Any, List, Tuple, Optional
from collections import deque
from .memory_interface import MemoryInterface, QueryContext, RetrievedMemory, MemoryData
import numpy as np
try:
    from sentence_transformers import SentenceTransformer, util  # Import the library
except ImportError:
    logging.error("Could not import sentence_transformers. Make sure it's installed in your environment")
    # These will be referenced later, so define placeholders
    SentenceTransformer = None
    util = None

# --- Embedding Model Initialization ---
# Load a pre-trained model. Choose one appropriate for your needs.
# 'all-MiniLM-L6-v2' is small and fast.
# Consider models like 'paraphrase-mpnet-base-v2' for potentially better quality.
# This should ideally be configurable or loaded based on config.
try:
    # Make this model loading potentially configurable via the main config dict
    embedding_model_name = 'all-MiniLM-L6-v2'
    embedding_model = SentenceTransformer(embedding_model_name)
    logging.info(f"SentenceTransformer model '{embedding_model_name}' loaded.")
    # Get embedding dimension dynamically
    EMBEDDING_DIM = embedding_model.get_sentence_embedding_dimension()
    logging.info(f"Embedding dimension: {EMBEDDING_DIM}")
except Exception as e:
    logging.error(f"Failed to load SentenceTransformer model: {e}. Embeddings will not work.", exc_info=True)
    embedding_model = None
    EMBEDDING_DIM = 0

# --- Embedding and Similarity Functions ---
def get_embedding(text: str) -> Optional[List[float]]:
    """Generates text embeddings using the loaded SentenceTransformer model."""
    if not text or embedding_model is None:
        return None
    try:
        # The model returns a numpy array, convert to list for consistency/serialization if needed
        embedding = embedding_model.encode(text, convert_to_numpy=True)
        return embedding.tolist() # Convert numpy array to list
    except Exception as e:
        logging.error(f"Error generating embedding for text '{text[:50]}...': {e}", exc_info=True)
        return None

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculates cosine similarity using sentence-transformers util."""
    if not vec1 or not vec2 or embedding_model is None:
        return 0.0
    try:
        # Convert lists back to tensors/numpy arrays if needed by util.cos_sim
        # util.cos_sim expects PyTorch tensors by default
        import torch # Ensure torch is imported
        tensor1 = torch.tensor(vec1).unsqueeze(0) # Add batch dimension
        tensor2 = torch.tensor(vec2).unsqueeze(0) # Add batch dimension
        similarity_tensor = util.cos_sim(tensor1, tensor2)
        return similarity_tensor.item() # Extract the scalar value
    except Exception as e:
        logging.error(f"Error calculating cosine similarity: {e}", exc_info=True)
        return 0.0


class EmotionalMemoryCore(MemoryInterface):
    """
    Stores experiences, facilitates RL, prediction, and evolutionary adaptation.
    Includes persistence and placeholders for advanced retrieval.
    """
    def __init__(self, config=None):
        if config is None:
            config = {}
        super().__init__(config)
        # Support both dict and dataclass configs
        _get = config.get if isinstance(config, dict) else lambda k, d=None: getattr(config, k, d)
        self.max_memory_size = _get('max_memory_size', 10000)
        self.persistence_path = _get('persistence_path', 'memory_state.pkl')
        self.memory_storage: deque[Tuple[float, MemoryData]] = deque(maxlen=self.max_memory_size)
        self._load_memory() # Load previous state if available
        logging.info(f"EmotionalMemoryCore initialized. Loaded {len(self.memory_storage)} memories from {self.persistence_path if os.path.exists(self.persistence_path) else 'new state'}. Max size: {self.max_memory_size}.")
        # TODO: Initialize vector database/index if using one.

    def _load_memory(self):
        """Loads memory state from the persistence path."""
        if self.persistence_path and os.path.exists(self.persistence_path):
            try:
                with open(self.persistence_path, 'rb') as f:
                    loaded_data = pickle.load(f)
                    if isinstance(loaded_data, deque):
                        # Ensure loaded data respects the current max_memory_size
                        if loaded_data.maxlen != self.max_memory_size:
                             logging.warning(f"Loaded memory maxlen ({loaded_data.maxlen}) differs from config ({self.max_memory_size}). Adjusting.")
                             # Create new deque with correct maxlen from loaded data
                             self.memory_storage = deque(loaded_data, maxlen=self.max_memory_size)
                        else:
                             self.memory_storage = loaded_data
                        logging.info(f"Successfully loaded {len(self.memory_storage)} memories.")
                    else:
                        logging.error(f"Failed to load memory: Expected deque, got {type(loaded_data)}")
            except (pickle.UnpicklingError, EOFError, FileNotFoundError, Exception) as e:
                logging.error(f"Error loading memory state from {self.persistence_path}: {e}", exc_info=True)
                # Start with an empty memory if loading fails
                self.memory_storage = deque(maxlen=self.max_memory_size)
        else:
            logging.info("No existing memory state found or persistence path not set. Starting fresh.")
            self.memory_storage = deque(maxlen=self.max_memory_size)

    def save_memory(self):
        """Saves the current memory state to the persistence path."""
        if self.persistence_path:
            try:
                # Create directory if it doesn't exist
                dir_name = os.path.dirname(self.persistence_path)
                if dir_name:
                    os.makedirs(dir_name, exist_ok=True)
                with open(self.persistence_path, 'wb') as f:
                    pickle.dump(self.memory_storage, f)
                logging.info(f"Successfully saved {len(self.memory_storage)} memories to {self.persistence_path}")
            except Exception as e:
                logging.error(f"Error saving memory state to {self.persistence_path}: {e}", exc_info=True)
        else:
            logging.warning("Persistence path not set. Memory not saved.")

    def store(self, timestamp: float, data: MemoryData):
        """
        Stores the experience data, ensuring essential fields for RL and context.
        Generates and stores text embeddings for relevant fields.
        """
        if not isinstance(data, dict):
             logging.error(f"Memory store failed: Data must be a dictionary, got {type(data)}")
             return

        # --- Enrich data for storage ---
        data.setdefault('state_summary', None)
        data.setdefault('action', None)
        data.setdefault('reward', None)
        data.setdefault('next_state_summary', None)
        data.setdefault('emotional_state', None)
        data.setdefault('active_goal', None)
        data.setdefault('active_lora_id', None)

        # Generate and store embeddings if model is available
        if embedding_model:
            state_text = data.get('state_summary', '')
            data['state_embedding'] = get_embedding(state_text) # Already returns list or None
        else:
            data['state_embedding'] = None # Ensure field exists even if model failed

        logging.debug(f"Storing memory at timestamp {timestamp} with embedding: {'Yes' if data['state_embedding'] else 'No'}")
        self.memory_storage.append((timestamp, data))

    def retrieve(self, query_context: QueryContext, top_k: int = 5) -> List[RetrievedMemory]:
        """
        Retrieves relevant memories using context (recency, semantic similarity, emotion).
        """
        logging.debug(f"Retrieving memories (top_k={top_k}) with context: {query_context}")
        if not self.memory_storage: return []

        # --- Prepare Query Context ---
        query_text = query_context.get("perception", {}).get("summary_text", "")
        query_embedding = get_embedding(query_text) if embedding_model else None # Generate embedding for query
        query_emotion = query_context.get("emotion", {})

        # --- Scoring Logic ---
        scored_memories = []
        current_time = time.time()
        search_pool_size = min(len(self.memory_storage), max(top_k * 10, 100))
        search_pool = list(self.memory_storage)[-search_pool_size:]

        for timestamp, data in reversed(search_pool):
            score = 0.0
            weight = 1.0

            # 1. Recency Score
            time_delta = current_time - timestamp
            recency_score = 1.0 / (1 + time_delta / 300.0)
            score += recency_score * 0.5

            # 2. Semantic Similarity Score (using REAL embeddings)
            memory_embedding = data.get('state_embedding')
            if query_embedding and memory_embedding:
                # Ensure embeddings are lists of floats before passing
                if isinstance(query_embedding, list) and isinstance(memory_embedding, list):
                     similarity = cosine_similarity(query_embedding, memory_embedding)
                     score += similarity * 1.0 # Weight similarity
                else:
                     logging.warning("Embeddings are not lists, skipping similarity calculation.")

            # 3. Emotional Similarity Score (Placeholder logic remains)
            memory_emotion = data.get('emotional_state')
            if query_emotion and memory_emotion and isinstance(memory_emotion, dict):
                emotion_distance = sum(abs(query_emotion.get(k, 0) - memory_emotion.get(k, 0)) for k in query_emotion)
                emotion_similarity = 1.0 / (1.0 + emotion_distance)
                score += emotion_similarity * 0.3

            # 4. Goal Relevance Score (Placeholder)
            # 5. Apply Meta-Memory Weight (Placeholder)
            final_score = score * weight
            scored_memories.append((final_score, timestamp, data))

        # Sort by score
        scored_memories.sort(key=lambda x: x[0], reverse=True)

        # Return top_k
        retrieved: List[RetrievedMemory] = [mem_data for score, ts, mem_data in scored_memories[:top_k]]
        logging.debug(f"Retrieved {len(retrieved)} memories based on context.")
        return retrieved

    def retrieve_batch_for_rl(self, batch_size: int) -> List[Tuple[float, MemoryData]]:
        """
        Retrieves a batch of recent experiences, typically used for RL updates.
        Returns tuples of (timestamp, data).
        """
        if batch_size > len(self.memory_storage):
            logging.warning(f"Requested RL batch size ({batch_size}) larger than memory size ({len(self.memory_storage)}). Returning all.")
            batch_size = len(self.memory_storage)

        if batch_size == 0:
            return []

        # Simple random sampling of recent experiences
        # More sophisticated sampling (e.g., Prioritized Experience Replay) can be added
        indices = np.random.choice(len(self.memory_storage), batch_size, replace=False)
        batch = [self.memory_storage[i] for i in indices]

        # Alternative: Just return the most recent batch_size items
        # batch = list(self.memory_storage)[-batch_size:]

        logging.debug(f"Retrieved batch of {len(batch)} memories for RL.")
        return batch

    # --- Optional methods ---
    # def _get_meta_weight(self, data: MemoryData) -> float:
    #     """Placeholder for calculating meta-memory weight."""
    #     # TODO: Implement logic based on reinforcement, surprise, outcome, etc.
    #     return 1.0

    # def check_coherence(self) -> float: ...
    # def query_survival_rate(self, last_n: int) -> float: ...

    def __del__(self):
        """Ensure memory is saved when the object is destroyed (e.g., program exit)."""
        logging.info("EmotionalMemoryCore shutting down. Saving memory...")
        self.save_memory()
