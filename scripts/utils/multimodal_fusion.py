from models.speech.whisper.whisper_integration import WhisperIntegration


class PaLI2Integration:
    """Stub for PaLI-2 vision-language model"""
    def generate_caption(self, image):
        return "placeholder caption"


class MultimodalFusion:
    def __init__(self):
        self.vision_model = PaLI2Integration()
        self.speech_model = WhisperIntegration()
        self.extra_modalities = {}

    def register_modality(self, name, model):
        self.extra_modalities[name] = model

    def fuse_inputs(self, image, audio_path, text, **extra_inputs):
        caption = self.vision_model.generate_caption(image)
        transcription = self.speech_model.transcribe_audio(audio_path)
        fused_data = {"caption": caption, "transcription": transcription, "text": text}

        for name, input_data in extra_inputs.items():
            if name in self.extra_modalities:
                fused_data[name] = self.extra_modalities[name].process(input_data)
        return fused_data