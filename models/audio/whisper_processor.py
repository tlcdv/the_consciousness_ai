import whisper
import torch

class WhisperProcessor:
    def __init__(self, config: dict):
        self.model = whisper.load_model("large-v3")
        self.emotion_classifier = self._load_emotion_classifier()
    
    def transcribe(self, audio: torch.Tensor) -> str:
        return self.model.transcribe(audio)["text"]
        
    def process_audio(self, audio: torch.Tensor) -> torch.Tensor:
        features = self.model.encode(audio)
        return self.emotion_classifier(features)