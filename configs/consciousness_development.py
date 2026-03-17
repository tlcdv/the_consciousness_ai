"""
Configuration for consciousness development stages.
Loads parameters from consciousness_development.yaml when available.
"""

from dataclasses import dataclass, field
import os
import logging

try:
    import yaml
except ImportError:
    yaml = None


@dataclass
class DevelopmentConfig:
    """Configuration for consciousness development stages and testing."""
    # Stage thresholds
    emergence_threshold: float = 0.5
    attention_threshold: float = 0.7
    emotional_learning_threshold: float = 0.6
    memory_coherence_threshold: float = 0.7
    self_awareness_threshold: float = 0.8

    # Testing parameters
    test_episodes: int = 10
    evaluation_window: int = 100
    min_stage_duration: int = 50

    # Survival metrics
    survival_metrics: dict = field(default_factory=lambda: {
        'stress_threshold': 0.6,
        'adaptation_rate': 0.01,
    })

    # Attention config
    attention: dict = field(default_factory=lambda: {
        'base_threshold': 0.5,
        'high_threshold': 0.8,
    })

    # Consciousness sub config (for nested access like config.consciousness.emergence_threshold)
    consciousness: 'DevelopmentConfig' = None

    def __post_init__(self):
        # Allow nested access: config.consciousness.emergence_threshold
        if self.consciousness is None:
            self.consciousness = self

    def __getitem__(self, key):
        """Allow dict-style access for compatibility."""
        return getattr(self, key)

    def get(self, key, default=None):
        """Allow dict-style .get() for compatibility."""
        return getattr(self, key, default)

    @classmethod
    def from_yaml(cls, path: str = None) -> 'DevelopmentConfig':
        """Load config from YAML file if available."""
        if path is None:
            path = os.path.join(
                os.path.dirname(__file__),
                'consciousness_development.yaml'
            )
        if yaml and os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    data = yaml.safe_load(f) or {}
                return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
            except Exception as e:
                logging.warning(f"Failed to load config from {path}: {e}")
        return cls()
