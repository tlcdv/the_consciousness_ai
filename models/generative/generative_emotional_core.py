"""
Generative Emotional Core

This module implements:
1. Emotion-aware text generation
2. Integration with LLaMA 3.3 for emotional context
3. Memory-guided generation
4. Emotional coherence validation

Dependencies:
- models/emotion/tgnn/emotional_graph.py for emotion processing
- models/memory/emotional_memory_core.py for memory access
- models/evaluation/consciousness_monitor.py for metrics
"""
from __future__ import annotations

import torch
import numpy as np
from dataclasses import dataclass
try:
    from transformers import LlamaTokenizer, LlamaForCausalLM
    _HAS_TRANSFORMERS = True
except ImportError:
    LlamaTokenizer = None
    LlamaForCausalLM = None
    _HAS_TRANSFORMERS = False
from models.emotion.tgnn.emotional_graph import EmotionalGraphNetwork
from models.memory.emotional_memory_core import EmotionalMemoryCore
import logging

@dataclass
class GenerativeConfig:
    """Configuration for generative emotional processing"""
    model_name: str = "llama-3.3"
    max_length: int = 1024
    temperature: float = 0.7
    emotional_weight: float = 0.8
    memory_weight: float = 0.6
    top_k_memories: int = 5

class GenerativeEmotionalCore:
    """
    Integrates generative AI with emotional memory for consciousness development
    
    Key Features:
    1. Emotional memory-conditioned generation
    2. Experience-based narrative creation
    3. Emotional context preservation
    4. Memory-guided response generation
    """
    
    def __init__(self, config=None):
        """Initialize generative emotional system"""
        if isinstance(config, dict):
            gen_cfg = config.get('generative_config', config)
            self.config = GenerativeConfig(
                model_name=gen_cfg.get('model_name', 'llama-3.3'),
                max_length=gen_cfg.get('max_length', 1024),
                temperature=gen_cfg.get('temperature', 0.7),
                emotional_weight=gen_cfg.get('emotional_weight', 0.8),
            )
        elif config is None:
            self.config = GenerativeConfig()
        else:
            self.config = config

        # Defer heavy model loading. Set to None until explicitly loaded.
        self.tokenizer = None
        self.model = None
        self.emotion_network = None
        self.memory_core = EmotionalMemoryCore(config if isinstance(config, dict) else {})
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def generate_response(self, prompt, emotional_context=None):
        """Generate emotionally aware response (lightweight fallback if model not loaded)."""
        if self.model is not None:
            return self.generate_with_emotion(prompt, emotional_context or {})
        # Fallback: return placeholder response
        response = f"[Generated response to: {prompt[:80] if isinstance(prompt, str) else 'tensor input'}]"
        metadata = {
            'emotional_context': emotional_context or {},
            'generation_mode': 'fallback',
        }
        return response, metadata
        
    def generate_with_emotion(
        self,
        prompt: str,
        emotional_context: dict[str, float],
        memory_context: list[dict] | None = None
    ) -> tuple[str, dict[str, float]]:
        """Generate text with emotional awareness"""
        # Process emotional context
        emotional_features = self.emotion_network.process(emotional_context)
        
        # Retrieve relevant memories
        if memory_context is None:
            memory_context = self.memory_core.retrieve_similar_memories(
                emotion_query=emotional_features,
                k=self.config.top_k_memories
            )
            
        # Build enhanced prompt
        enhanced_prompt = self._build_emotional_prompt(
            prompt,
            emotional_features,
            memory_context
        )
        
        # Generate response
        generated_ids = self.model.generate(
            input_ids=enhanced_prompt["input_ids"].to(self.device),
            attention_mask=enhanced_prompt["attention_mask"].to(self.device),
            max_length=self.config.max_length,
            temperature=self.config.temperature,
            pad_token_id=self.tokenizer.eos_token_id,
            num_return_sequences=1
        )
        
        response = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        return response, {
            'emotional_coherence': self._evaluate_coherence(response, emotional_features),
            'memory_influence': len(memory_context) > 0,
            'generation_confidence': self.model.get_confidence()
        }
        
    def _build_emotional_prompt(
        self,
        prompt: str,
        emotional_features: torch.Tensor,
        memory_context: list[dict]
    ) -> dict:
        """Prepare context for generation with emotional conditioning"""
        
        # Create memory context string
        memory_context_str = self._format_memory_context(memory_context)
        
        # Create emotional prefix
        emotional_prefix = self._create_emotional_prefix(emotional_features)
        
        # Combine context elements
        full_context = f"{emotional_prefix}\n{memory_context_str}\n\nPrompt: {prompt}\nResponse:"
        
        # Tokenize
        tokenized = self.tokenizer(
            full_context,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        return tokenized
        
    def _format_memory_context(self, memories: list[dict]) -> str:
        """Format memories into context string"""
        context_parts = []
        
        for memory in memories:
            context_parts.append(
                f"Previous experience ({memory['emotion_values']['valence']:.2f} valence): {memory['narrative']}"
            )
            
        return "\n".join(context_parts)
        
    def _create_emotional_prefix(self, emotional_embedding: torch.Tensor) -> str:
        """Create emotional conditioning prefix"""
        # Project emotional embedding to text space
        emotional_projection = self.model.get_input_embeddings()(
            emotional_embedding.unsqueeze(0)
        )
        
        # Generate emotional context tokens
        emotional_tokens = self.model.generate(
            inputs_embeds=emotional_projection,
            max_length=50,
            temperature=0.5,
            num_return_sequences=1
        )
        
        return self.tokenizer.decode(emotional_tokens[0], skip_special_tokens=True)
        
    def _evaluate_coherence(self, response: str, emotional_features: torch.Tensor) -> float:
        """Evaluate emotional coherence of the response"""
        # Placeholder for actual coherence evaluation logic
        return 1.0
        
    def _store_interaction_memory(
        self,
        prompt: str,
        response: str,
        emotional_context: dict[str, float],
        situation_context: dict | None
    ):
        """Store interaction in emotional memory"""
        self.memory_core.store_experience({
            'prompt': prompt,
            'response': response,
            'emotion_values': emotional_context,
            'context': situation_context,
            'timestamp': np.datetime64('now')
        })
        
    def _get_generation_metadata(
        self,
        context: dict,
        response: str,
        emotional_context: dict[str, float]
    ) -> dict:
        """Get metadata about the generation process"""
        return {
            'context_length': len(context['input_ids'][0]),
            'response_length': len(response.split()),
            'emotional_context': emotional_context,
            'generation_timestamp': np.datetime64('now')
        }