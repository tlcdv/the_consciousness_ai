"""
Test suite for the Narrative Engine component of ACM.

Validates narrative generation and integration with memory/emotion dependencies.
"""
import unittest
from models.narrative.narrative_engine import NarrativeEngine


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
        self.narrative_engine = NarrativeEngine(
            foundational_model=mock_model,
            memory=mock_memory,
            emotion=mock_emotion,
            llm=mock_model,
        )

    def test_narrative_generation(self):
        """Test narrative generation with emotional context"""
        input_text = "The agent encountered a stressful situation"
        narrative = self.narrative_engine.generate_narrative(input_text)

        self.assertIsNotNone(narrative)
        self.assertTrue(len(narrative) > 0)
        self.assertIn('stress', narrative.lower())


if __name__ == "__main__":
    unittest.main()
