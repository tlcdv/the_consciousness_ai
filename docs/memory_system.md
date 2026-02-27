# Emotional Memory System (`EmotionalMemoryCore`)

## 1. Purpose and Role

The `EmotionalMemoryCore` is a crucial component of the The Consciousness AI. It serves as the agent's long-term storage for experiences, playing a vital role in:

*   **Learning:** Providing data for Reinforcement Learning (RL) algorithms (like Dreamer).
*   **Contextual Decision Making:** Retrieving relevant past experiences to inform current actions and predictions ("thought process").
*   **Adaptation:** Storing context about which model adaptations (e.g., LoRAs) were active during specific experiences.
*   **Cross-Simulation Development:** Persisting memories across simulation runs, allowing the agent to "inherit" knowledge and evolve over multiple "lifetimes".
*   **Emotional Grounding:** Associating experiences with the emotional states felt at the time, enabling emotionally relevant retrieval.

## 2. Core Features

*   **Chronological Storage:** Experiences are stored with timestamps in the order they occur.
*   **Limited Capacity:** Uses a `deque` with a configurable maximum size (`max_memory_size`) to keep memory bounded, automatically discarding the oldest entries.
*   **Persistence:** Loads existing memory from a file (`persistence_path`) on initialization and saves the current state on shutdown (`__del__`) using `pickle`.
*   **Rich Data Storage:** Stores comprehensive `MemoryData` dictionaries including state/perception summaries, actions, rewards, emotional states, active goals, and model adaptation context (e.g., `active_lora_id`).
*   **Embedding Generation:** Includes placeholder logic to generate vector embeddings for text summaries (e.g., `state_summary`) during storage.
*   **Contextual Retrieval (`retrieve`):** Retrieves the `top_k` most relevant memories based on a `QueryContext` using a scoring system that considers:
    *   Recency (time decay).
    *   Semantic Similarity (placeholder cosine similarity between query text embedding and stored state embeddings).
    *   Emotional Similarity (placeholder distance calculation between query emotion and stored emotions).
*   **RL Batch Retrieval (`retrieve_batch_for_rl`):** Provides a method to sample a random batch of experiences, suitable for training RL models.
*   **Configuration:** Key parameters like `max_memory_size` and `persistence_path` are configurable.

## 3. Data Structure (`MemoryData` Dictionary)

Each memory entry stored is a dictionary (`MemoryData`) associated with a timestamp. Key fields include:

*   `timestamp` (float): Time the experience occurred (external to the dict, stored alongside it).
*   `state_summary` (Optional[str]): Textual summary of the agent's perception/state.
*   `action` (Optional[Dict]): The action taken by the agent.
*   `reward` (Optional[float]): The combined reward received (environmental + internal/emotional).
*   `next_state_summary` (Optional[str]): Textual summary of the resulting state/perception.
*   `emotional_state` (Optional[Dict[str, float]]): The agent's emotional state during the experience.
*   `active_goal` (Optional[Any]): Identifier or description of the agent's goal at the time.
*   `active_lora_id` (Optional[str]): Identifier for the specific model adaptation (e.g., LoRA) active during the experience.
*   `state_embedding` (Optional[List[float]]): Vector embedding generated from `state_summary`.
*   *(Other relevant data from `ConsciousnessCore`'s integrated state can be added)*

## 4. Storage Mechanism

*   Uses `collections.deque(maxlen=max_memory_size)` for efficient appending and automatic discarding of the oldest memories when the maximum size is reached.

## 5. Persistence

*   **Loading:** The `_load_memory` method attempts to load a previously saved `deque` from the file specified by `config['persistence_path']` (default: `memory_state.pkl`) during initialization. Handles potential errors during loading.
*   **Saving:** The `save_memory` method saves the current `deque` to the persistence path. This is automatically called by the `__del__` method when the `EmotionalMemoryCore` object is garbage collected (typically on program exit).
*   **Format:** Currently uses Python's `pickle` module.
    *   **Warning:** `pickle` is not secure against maliciously crafted data. For production or sharing memory files, consider safer alternatives like `jsonlines` (saving each memory entry as a JSON object on a new line) or a dedicated database (like SQLite, DuckDB, or a vector database).
*   **Directory Creation:** Automatically creates the directory for the persistence path if it doesn't exist.

## 6. Contextual Retrieval (`retrieve` method)

*   **Input:** Takes a `QueryContext` dictionary (containing current perception, emotion, goal, etc.) and `top_k`.
*   **Process:**
    1.  Retrieves a pool of recent memories.
    2.  Generates an embedding from the query text (if available).
    3.  Iterates through the pool, calculating a score for each memory based on:
        *   **Recency:** Decays over time.
        *   **Semantic Similarity:** Cosine similarity between query embedding and memory's `state_embedding` (placeholder).
        *   **Emotional Similarity:** Inverse distance between query `emotional_state` and memory's `emotional_state` (placeholder).
        *   *(Goal Relevance scoring is planned)*.
        *   *(Meta-Memory weighting is planned)*.
    4.  Sorts memories by score (highest first).
    5.  Returns the `MemoryData` dictionaries of the `top_k` highest-scoring memories.
*   **Purpose:** To provide relevant past context for the agent's current processing cycle (e.g., for prediction, reasoning, or action selection).

## 7. RL Batch Retrieval (`retrieve_batch_for_rl` method)

*   **Input:** Takes `batch_size`.
*   **Process:** Currently implements simple random sampling without replacement from the entire memory deque using `numpy.random.choice`.
*   **Output:** Returns a list of `(timestamp, MemoryData)` tuples.
*   **Purpose:** To provide batches of experiences suitable for training RL algorithms.
*   **Future:** Could be enhanced with techniques like Prioritized Experience Replay (PER).

## 8. Embeddings

*   Includes placeholder functions `get_embedding` and `cosine_similarity`.
*   Currently generates simple, deterministic "embeddings" based on hash values for demonstration.
*   **TODO:** Replace placeholders with a real sentence embedding model (e.g., from the `sentence-transformers` library) and potentially integrate with a vector database (like FAISS, ChromaDB, LanceDB) for efficient similarity search, especially as memory size grows.

## 9. Adaptation Context (`active_lora_id`)

*   The `store` method includes `setdefault('active_lora_id', None)`.
*   This allows associating specific experiences with the model adaptation (like a LoRA layer) that was active at the time.
*   **Use Cases:**
    *   Analyzing the performance of different adaptations.
    *   Retrieving memories relevant to a specific adaptation during fine-tuning or inference.
    *   Potentially weighting memory retrieval based on the currently active adaptation.

## 10. Configuration

The `__init__` method accepts a `config` dictionary with keys:

*   `max_memory_size` (int, default: 10000): Maximum number of memories to store.
*   `persistence_path` (str, default: 'memory_state.pkl'): Path to the file for loading/saving memory.

## 11. Future Work / TODOs

*   Replace placeholder embedding/similarity functions with real implementations.
*   Integrate a vector database/index for efficient semantic search.
*   Implement more sophisticated retrieval scoring (goal relevance, meta-memory weights based on outcome/surprise).
*   Implement advanced RL sampling strategies (e.g., PER).
*   Consider alternative, safer persistence formats (e.g., `jsonlines`, database).
*   Implement `check_coherence` and `query_survival_rate` methods if needed.
*   Add more robust error handling for persistence.