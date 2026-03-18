"""
Narrative Engine for The Consciousness AI

LLM-backed narrative generation with memory retrieval, emotional context
injection, and coherence tracking. Gracefully degrades to template-based
generation when LLM weights are unavailable.

Integration sites:
- simulations/api/simulation_manager.py
- models/controller/simulation_controller.py
- tests/test_emotional_reinforcement_success.py
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

# Attempt to load HuggingFace transformers for LLM-backed generation
_HF_AVAILABLE = False
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    _HF_AVAILABLE = True
except ImportError:
    logger.info("transformers not available, NarrativeEngine will use template fallback")


@dataclass
class NarrativeResult:
    """Output of a narrative generation call."""
    text: str
    coherence: float = 1.0
    method: str = "template"  # "llm", "template", or "injected"
    emotional_context: dict = field(default_factory=dict)


class CoherenceTracker:
    """Tracks narrative coherence across a rolling window using keyword overlap."""

    def __init__(self, capacity: int = 20):
        self.capacity = capacity
        self._history: deque[set[str]] = deque(maxlen=capacity)

    def update(self, text: str) -> float:
        """Add a narrative and return coherence score against recent history."""
        words = set(text.lower().split())
        if not self._history:
            self._history.append(words)
            return 1.0

        # Jaccard similarity with the union of recent narratives
        recent_union = set()
        for s in self._history:
            recent_union |= s
        if not recent_union or not words:
            self._history.append(words)
            return 0.0

        overlap = len(words & recent_union) / len(words | recent_union)
        self._history.append(words)
        return float(overlap)

    def get_coherence(self) -> float:
        """Return average pairwise coherence of the last few entries."""
        if len(self._history) < 2:
            return 1.0
        recent = list(self._history)[-5:]
        scores = []
        for i in range(len(recent)):
            for j in range(i + 1, len(recent)):
                union = recent[i] | recent[j]
                if union:
                    scores.append(len(recent[i] & recent[j]) / len(union))
        return float(np.mean(scores)) if scores else 1.0


class NarrativeEngine:
    """
    LLM-backed narrative engine with graceful degradation.

    Constructor signature preserved for backward compatibility:
        NarrativeEngine(foundational_model, memory, emotion, llm)

    When `llm` is a real HuggingFace model name (str) or pipeline, the engine
    uses it for generation. Otherwise it falls back to template-based output
    using the injected mock/stub objects.
    """

    # Default model for LLM-backed generation (small enough for CPU)
    DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B"

    def __init__(
        self,
        foundational_model=None,
        memory=None,
        emotion=None,
        llm=None,
        *,
        model_id: str | None = None,
        max_new_tokens: int = 128,
        coherence_window: int = 20,
    ):
        self.foundational_model = foundational_model
        self.memory = memory
        self.emotion = emotion
        self.llm = llm

        # Narrative state
        self.memory_context: list[str] = []
        self.current_narrative_text = ""
        self.narrative_history: list[dict] = []
        self.current_context: dict = {}

        # Coherence tracking
        self.coherence_tracker = CoherenceTracker(capacity=coherence_window)

        # LLM setup
        self.max_new_tokens = max_new_tokens
        self._tokenizer = None
        self._model = None
        self._llm_ready = False

        if model_id is not None:
            self._init_llm(model_id)

    def _init_llm(self, model_id: str) -> None:
        """Load a HuggingFace causal LM. Fails gracefully if unavailable."""
        if not _HF_AVAILABLE:
            logger.warning("transformers not installed, skipping LLM init")
            return
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_id, trust_remote_code=True
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                model_id, trust_remote_code=True
            )
            self._model.eval()
            self._llm_ready = True
            logger.info("NarrativeEngine LLM loaded: %s", model_id)
        except Exception as e:
            logger.warning("Failed to load LLM %s: %s. Using template fallback.", model_id, e)
            self._llm_ready = False

    def _generate_with_llm(self, prompt: str) -> str:
        """Generate text using the loaded HuggingFace model."""
        import torch

        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        # Decode only the generated tokens (skip the prompt)
        generated = output_ids[0, inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(generated, skip_special_tokens=True).strip()

    # --- Template fallback ---

    _TEMPLATES = {
        "high_arousal_positive": "Energized by {subject}, feeling driven to act.",
        "low_arousal_positive": "Calmly processing {subject}, a settled awareness.",
        "high_arousal_negative": "Alarmed by {subject}, urgent attention required.",
        "low_arousal_negative": "Low energy regarding {subject}, a dull unease.",
        "neutral": "Noting {subject} without strong affect.",
    }

    @staticmethod
    def _pad_quadrant(valence: float, arousal: float) -> str:
        if abs(valence) < 0.3 and arousal < 0.5:
            return "neutral"
        if valence >= 0.3:
            return "high_arousal_positive" if arousal >= 0.5 else "low_arousal_positive"
        return "high_arousal_negative" if arousal >= 0.5 else "low_arousal_negative"

    def _generate_template(self, subject: str, emotional_state: dict) -> str:
        quadrant = self._pad_quadrant(
            emotional_state.get("valence", 0.0),
            emotional_state.get("arousal", 0.0),
        )
        return self._TEMPLATES[quadrant].format(subject=subject)

    # --- Public API ---

    def update_narrative(self, chain_text: str) -> None:
        """Update the agent's internal narrative with a chain-of-thought string."""
        self.current_narrative_text = chain_text
        self.coherence_tracker.update(chain_text)

    def current_narrative(self) -> str:
        return self.current_narrative_text

    def generate_narrative(self, input_text: str, emotional_state: dict | None = None) -> NarrativeResult:
        """
        Generate a narrative from input text, integrating memory and emotion.

        Returns NarrativeResult with text, coherence score, and method used.
        For backward compat, also works when called with just input_text
        and injected memory/emotion dependencies.
        """
        # Retrieve memories
        memories_str = ""
        if self.memory is not None:
            try:
                memories_str = str(self.memory.retrieve_relevant(input_text))
            except Exception:
                memories_str = ""

        # Get emotional context
        emotion_str = ""
        if emotional_state is not None:
            emotion_str = f"valence={emotional_state.get('valence', 0):.2f}, arousal={emotional_state.get('arousal', 0):.2f}, dominance={emotional_state.get('dominance', 0):.2f}"
        elif self.emotion is not None:
            try:
                emotion_str = str(self.emotion.analyze(input_text))
            except Exception:
                emotion_str = ""

        # Build prompt
        prompt = self._build_prompt(input_text, memories_str, emotion_str)

        # Generate
        if self._llm_ready:
            text = self._generate_with_llm(prompt)
            method = "llm"
        elif self.llm is not None and hasattr(self.llm, "generate"):
            # Use injected llm dependency (mock or real)
            text = self.llm.generate(prompt)
            method = "injected"
        else:
            text = self._generate_template(
                input_text,
                emotional_state or {"valence": 0.0, "arousal": 0.0},
            )
            method = "template"

        # Track coherence
        coherence = self.coherence_tracker.update(text)
        self.current_narrative_text = text
        self.memory_context.append(text)

        return NarrativeResult(
            text=text,
            coherence=coherence,
            method=method,
            emotional_context=emotional_state or {},
        )

    def generate_self_reflection(self, interaction_log: list) -> str:
        """Generate a reflective narrative based on past interactions."""
        refined_log = "\n".join([str(entry) for entry in interaction_log[-10:]])
        prompt = f"Reflect on these interactions:\n{refined_log}"

        if self._llm_ready:
            text = self._generate_with_llm(prompt)
        elif self.foundational_model is not None and hasattr(self.foundational_model, "generate"):
            text = self.foundational_model.generate(prompt)
        else:
            text = f"Reflecting on {len(interaction_log)} past interactions."

        self.current_narrative_text = text
        self.coherence_tracker.update(text)
        return text

    def integrate_experience(self, experience: dict) -> None:
        """Integrate new experience into narrative context."""
        self.narrative_history.append(experience)
        self.current_context = self._update_context(experience)

    def get_coherence(self) -> float:
        """Return current narrative coherence score."""
        return self.coherence_tracker.get_coherence()

    def _build_prompt(self, input_text: str, memories: str, emotional_context: str) -> str:
        parts = [f"Current situation: {input_text}"]
        if memories:
            parts.append(f"Relevant memories: {memories}")
        if emotional_context:
            parts.append(f"Emotional state: {emotional_context}")
        parts.append("Generate a first-person narrative reflection:")
        return "\n".join(parts)

    def _update_context(self, new_experience: dict) -> dict:
        ctx = dict(self.current_context)
        ctx.update(new_experience)
        return ctx
