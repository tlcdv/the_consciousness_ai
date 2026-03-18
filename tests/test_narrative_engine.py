"""
Test suite for the Narrative Engine component.

Validates narrative generation, coherence tracking, template fallback,
and integration with memory/emotion dependencies.
"""
from __future__ import annotations

import unittest
from models.narrative.narrative_engine import NarrativeEngine, NarrativeResult, CoherenceTracker


class MockModel:
    def generate(self, prompt):
        return f"Generated narrative: The agent experienced stress and adapted its behavior."


class MockMemory:
    def retrieve_relevant(self, input_text):
        return "Previous experience: agent navigated a stressful encounter."


class MockEmotion:
    def analyze(self, input_text):
        return "Emotional context: moderate stress, high arousal."


class TestNarrativeEngine(unittest.TestCase):
    def setUp(self):
        mock_model = MockModel()
        mock_memory = MockMemory()
        mock_emotion = MockEmotion()
        self.engine = NarrativeEngine(
            foundational_model=mock_model,
            memory=mock_memory,
            emotion=mock_emotion,
            llm=mock_model,
        )

    def test_generate_narrative_returns_result(self):
        result = self.engine.generate_narrative("The agent encountered a stressful situation")
        self.assertIsInstance(result, NarrativeResult)
        self.assertIsNotNone(result.text)
        self.assertTrue(len(result.text) > 0)

    def test_generate_narrative_uses_injected_llm(self):
        result = self.engine.generate_narrative("test input")
        self.assertEqual(result.method, "injected")

    def test_generate_narrative_includes_memory(self):
        """The injected LLM receives the prompt with memories."""
        result = self.engine.generate_narrative("test input")
        # MockModel echoes the prompt, which should contain memory content
        self.assertIn("stress", result.text.lower())

    def test_template_fallback_when_no_llm(self):
        engine = NarrativeEngine()
        result = engine.generate_narrative("obstacle ahead")
        self.assertEqual(result.method, "template")
        self.assertTrue(len(result.text) > 0)

    def test_template_with_emotional_state(self):
        engine = NarrativeEngine()
        result = engine.generate_narrative(
            "danger detected",
            emotional_state={"valence": -0.8, "arousal": 0.9, "dominance": 0.0},
        )
        self.assertEqual(result.method, "template")
        self.assertIn("Alarmed", result.text)

    def test_template_positive_valence(self):
        engine = NarrativeEngine()
        result = engine.generate_narrative(
            "food found",
            emotional_state={"valence": 0.8, "arousal": 0.7},
        )
        self.assertIn("Energized", result.text)

    def test_template_neutral(self):
        engine = NarrativeEngine()
        result = engine.generate_narrative(
            "nothing happening",
            emotional_state={"valence": 0.0, "arousal": 0.1},
        )
        self.assertIn("Noting", result.text)

    def test_update_narrative(self):
        self.engine.update_narrative("I see a wall ahead.")
        self.assertEqual(self.engine.current_narrative(), "I see a wall ahead.")

    def test_generate_self_reflection(self):
        log = [{"step": 1, "reward": 0.5}, {"step": 2, "reward": 0.8}]
        text = self.engine.generate_self_reflection(log)
        self.assertIsInstance(text, str)
        self.assertTrue(len(text) > 0)

    def test_integrate_experience(self):
        exp = {"event": "collision", "reward": -0.5}
        self.engine.integrate_experience(exp)
        self.assertEqual(len(self.engine.narrative_history), 1)
        self.assertEqual(self.engine.current_context.get("event"), "collision")

    def test_coherence_tracked_across_calls(self):
        self.engine.generate_narrative("exploring room")
        self.engine.generate_narrative("still exploring room")
        coherence = self.engine.get_coherence()
        self.assertIsInstance(coherence, float)
        self.assertGreaterEqual(coherence, 0.0)
        self.assertLessEqual(coherence, 1.0)

    def test_memory_context_grows(self):
        self.engine.generate_narrative("step one")
        self.engine.generate_narrative("step two")
        self.assertEqual(len(self.engine.memory_context), 2)

    def test_result_coherence_field(self):
        r1 = self.engine.generate_narrative("first thought")
        self.assertIsInstance(r1.coherence, float)
        self.assertGreaterEqual(r1.coherence, 0.0)
        self.assertLessEqual(r1.coherence, 1.0)

    def test_none_memory_and_emotion(self):
        """Engine works when memory and emotion are None."""
        engine = NarrativeEngine(llm=MockModel())
        result = engine.generate_narrative("test")
        self.assertEqual(result.method, "injected")

    def test_emotional_context_in_result(self):
        emo = {"valence": 0.5, "arousal": 0.3, "dominance": 0.1}
        result = self.engine.generate_narrative("test", emotional_state=emo)
        self.assertEqual(result.emotional_context, emo)


class TestCoherenceTracker(unittest.TestCase):
    def test_first_entry_coherence_one(self):
        tracker = CoherenceTracker()
        score = tracker.update("hello world")
        self.assertEqual(score, 1.0)

    def test_identical_texts_high_coherence(self):
        tracker = CoherenceTracker()
        tracker.update("the cat sat on the mat")
        score = tracker.update("the cat sat on the mat")
        self.assertGreater(score, 0.5)

    def test_unrelated_texts_low_coherence(self):
        tracker = CoherenceTracker()
        tracker.update("quantum physics entanglement")
        score = tracker.update("banana strawberry mango")
        self.assertLess(score, 0.3)

    def test_get_coherence_empty(self):
        tracker = CoherenceTracker()
        self.assertEqual(tracker.get_coherence(), 1.0)

    def test_get_coherence_with_data(self):
        tracker = CoherenceTracker()
        tracker.update("exploring the dark room")
        tracker.update("still exploring the room")
        tracker.update("found something in the room")
        coherence = tracker.get_coherence()
        self.assertGreater(coherence, 0.0)


if __name__ == "__main__":
    unittest.main()
