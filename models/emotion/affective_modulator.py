"""
Affective Modulator (Tier 2 Architecture)

Redesigns the emotion system from workspace competitor to parallel modulator.
Biological basis: the limbic system does not compete with sensory cortices
for conscious access. It modulates sensory processing via two mechanisms:

1. Valence Field: emotional valence biases sensory bid values.
   Positive valence boosts approach-relevant bids, negative valence
   boosts threat-relevant bids.

2. Arousal-Threshold Coupling: global arousal modulates the GNW ignition
   threshold. High arousal lowers the threshold (easier ignition),
   matching biological fight-or-flight heightened awareness.

Reference: Feinberg & Mallatt (2016), The Ancient Origins of Consciousness.
"""
from __future__ import annotations



# Semantic categories for bid modulation by valence
APPROACH_MODULES = {"vision", "audio", "memory", "body"}
THREAT_MODULES = {"body", "vision", "audio"}


class AffectiveModulator:
    """
    Parallel affective modulation layer for the Global Workspace.

    Receives PAD state and applies two modulation mechanisms:
    - Valence field modulation on specialist bids
    - Arousal-threshold coupling on GNW ignition threshold
    """

    def __init__(self, config: dict = None):
        config = config or {}

        # Valence field strength: how much valence influences bids
        self.valence_gain = config.get("valence_gain", 0.15)

        # Arousal coupling strength: how much arousal shifts ignition threshold
        self.arousal_gain = config.get("arousal_gain", 0.2)

        # Baseline ignition threshold (matches GNW default)
        self.baseline_threshold = config.get("baseline_threshold", 0.6)

        # Arousal target for homeostatic computation
        self.arousal_target = config.get("arousal_target", 0.3)

        # Dominance influence: high dominance slightly boosts all bids
        # (agent with strong sense of agency processes more actively)
        self.dominance_gain = config.get("dominance_gain", 0.05)

        # Interoceptive gain: how strongly homeostatic imbalance affects PAD
        self.intero_gain = config.get("intero_gain", 0.4)

    def interoceptive_to_pad(
        self,
        interoceptive_state: dict[str, float],
    ) -> dict[str, float]:
        """
        Convert interoceptive drives into PAD deltas.

        Biological basis: homeostatic imbalance generates affect independently
        of external stimuli. An agent with depleted energy feels negative
        valence even in a safe environment (Craig 2009, Damasio 1999).

        Mapping:
            energy:  low energy → negative valence (discomfort from depletion)
            fatigue: high fatigue → negative valence, reduced arousal (sluggishness)
            damage:  high damage → strong negative valence, high arousal (pain/alarm)

        Args:
            interoceptive_state: {"energy": float, "fatigue": float, "damage": float}
                All values in [0, 1].

        Returns:
            PAD delta dict {"valence": float, "arousal": float, "dominance": float}
        """
        energy = interoceptive_state.get("energy", 1.0)
        fatigue = interoceptive_state.get("fatigue", 0.0)
        damage = interoceptive_state.get("damage", 0.0)

        # Energy depletion: valence drops linearly below 0.5 energy
        energy_valence = min(0.0, (energy - 0.5) * self.intero_gain)

        # Fatigue: negative valence and suppressed arousal
        fatigue_valence = -fatigue * self.intero_gain * 0.5
        fatigue_arousal = -fatigue * self.intero_gain * 0.8

        # Damage: strong negative valence, arousal spike (pain alarm)
        damage_valence = -damage * self.intero_gain * 2.0
        damage_arousal = damage * self.intero_gain * 1.5

        # Damage also reduces dominance (feeling vulnerable)
        damage_dominance = -damage * self.intero_gain

        return {
            "valence": max(-1.0, energy_valence + fatigue_valence + damage_valence),
            "arousal": max(-1.0, min(1.0, fatigue_arousal + damage_arousal)),
            "dominance": max(-1.0, damage_dominance),
        }

    def modulate(
        self,
        bids: dict[str, float],
        pad_state: dict[str, float],
        interoceptive_state: dict[str, float] | None = None,
    ) -> tuple[dict[str, float], float]:
        """
        Apply affective modulation to workspace bids and ignition threshold.

        Args:
            bids: Specialist module bid values {name: float}
            pad_state: Current PAD state {"valence": float, "arousal": float, "dominance": float}
            interoceptive_state: Optional internal drives {"energy": float, "fatigue": float, "damage": float}.
                When provided, homeostatic imbalance generates additional PAD signals
                that are summed with the external PAD state before modulation.

        Returns:
            tuple of (modulated_bids, adjusted_ignition_threshold)
        """
        valence = pad_state.get("valence", 0.0)
        arousal = pad_state.get("arousal", 0.0)
        dominance = pad_state.get("dominance", 0.0)

        # Integrate interoceptive drives if available
        if interoceptive_state is not None:
            intero_pad = self.interoceptive_to_pad(interoceptive_state)
            valence = max(-1.0, min(1.0, valence + intero_pad["valence"]))
            arousal = max(-1.0, min(1.0, arousal + intero_pad["arousal"]))
            dominance = max(-1.0, min(1.0, dominance + intero_pad["dominance"]))

        # --- 1. Valence Field Modulation ---
        modulated_bids = {}
        for name, bid in bids.items():
            delta = 0.0

            if valence > 0:
                # Positive valence: boost approach-relevant modules
                if name in APPROACH_MODULES:
                    delta = valence * self.valence_gain
            elif valence < 0:
                # Negative valence: boost threat-relevant modules
                if name in THREAT_MODULES:
                    delta = abs(valence) * self.valence_gain

            # Dominance: high dominance slightly raises all bids
            # (active agency = more processing)
            delta += max(0.0, dominance) * self.dominance_gain

            modulated_bids[name] = max(0.0, min(1.0, bid + delta))

        # --- 2. Arousal-Threshold Coupling ---
        # High arousal -> lower threshold (easier ignition, heightened awareness)
        # Low arousal -> higher threshold (harder ignition, calm filtering)
        # Arousal is in [-1, 1]. We shift threshold down for positive arousal.
        threshold_shift = arousal * self.arousal_gain
        adjusted_threshold = self.baseline_threshold - threshold_shift

        # Clamp threshold to reasonable range
        adjusted_threshold = max(0.2, min(0.9, adjusted_threshold))

        return modulated_bids, adjusted_threshold
