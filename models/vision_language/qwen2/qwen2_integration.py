from __future__ import annotations

import torch
from typing import Any
import logging

logger = logging.getLogger(__name__)

try:
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info
    _QWEN2_AVAILABLE = True
except ImportError:
    _QWEN2_AVAILABLE = False
    logger.warning("Qwen2-VL dependencies not installed. Visual embedding will return zero tensors.")


# Qwen2-VL ViT hidden dimension (consistent across 2B, 7B, 72B variants)
_QWEN2_VIT_DIM = 1536


class Qwen2VLIntegration:
    """
    Integration for Qwen2-VL-7B-Instruct.

    Handles scene analysis and visual embedding extraction for the consciousness pipeline.
    Visual embeddings are extracted from the ViT encoder's last hidden state (mean-pooled),
    following the same approach as Qwen3-VL-Embedding.

    Supports 4-bit quantization via bitsandbytes. Falls back gracefully when model
    weights are unavailable (returns zero embeddings so tests can run without weights).
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.model_name = config.get("model_name", "Qwen/Qwen2-VL-7B-Instruct")
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")

        self.processor = None
        self.model = None

        self._load_model()

    def _load_model(self):
        """Load model and processor. Logs a warning and continues if weights are unavailable."""
        if not _QWEN2_AVAILABLE:
            logger.warning("Qwen2-VL not available. Running in stub mode.")
            return

        logger.info(f"Loading Qwen2-VL model: {self.model_name}")

        quantization_config = None
        if self.config.get("quantization", {}).get("load_in_4bit", False):
            from transformers import BitsAndBytesConfig
            q_cfg = self.config["quantization"]
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=q_cfg.get("bnb_4bit_quant_type", "nf4"),
                bnb_4bit_compute_dtype=getattr(torch, q_cfg.get("bnb_4bit_compute_dtype", "float16"))
            )
            logger.info("4-bit quantization enabled.")

        try:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                quantization_config=quantization_config,
                device_map="auto" if quantization_config else None,
                trust_remote_code=True
            )
            self.processor = AutoProcessor.from_pretrained(self.model_name)

            if not quantization_config:
                self.model.to(self.device)

            self.model.eval()
            logger.info("Qwen2-VL loaded successfully.")

        except Exception as e:
            logger.warning(f"Qwen2-VL weights not available, running in stub mode: {e}")
            self.model = None
            self.processor = None

    def analyze_scene(self, image_input: Any, prompt: str = "Describe this scene in detail.") -> str:
        """
        Analyze an image with a text prompt.

        Args:
            image_input: PIL Image, file path, URL, or base64 string.
            prompt: Text prompt for the analysis.

        Returns:
            Generated text description, or empty string in stub mode.
        """
        if self.model is None or self.processor is None:
            return ""

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_input},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        gen_config = self.config.get("generation", {})
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=gen_config.get("max_new_tokens", 128),
                temperature=gen_config.get("temperature", 0.7),
                top_p=gen_config.get("top_p", 0.9)
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0]

    def get_visual_embeddings(self, image_input: Any, return_spatial_grid: bool = False) -> torch.Tensor:
        """
        Extract visual feature embeddings from the ViT encoder.

        By default, returns a mean-pooled embedding over all visual tokens (1D tensor).
        If return_spatial_grid=True, reshapes the tokens into a 2D spatial grid [C, H, W]
        for use in topographically-aware modules like the Sensory Tectum.

        Args:
            image_input: PIL Image, file path, URL, or base64-encoded image.
            return_spatial_grid: If True, returns unpooled [C, H, W] tensor.

        Returns:
            Tensor of shape (1536,) or (1536, H, W). Zero tensor when model is not loaded.
        """
        if self.model is None or self.processor is None:
            if return_spatial_grid:
                # Stub grid shape, assume 14x14 patches as a fallback
                return torch.zeros(_QWEN2_VIT_DIM, 14, 14)
            return torch.zeros(_QWEN2_VIT_DIM)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_input},
                    {"type": "text", "text": ""},
                ],
            }
        ]

        try:
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            image_inputs, _ = process_vision_info(messages)

            inputs = self.processor(
                text=[text],
                images=image_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            pixel_values = inputs.get("pixel_values")
            image_grid_thw = inputs.get("image_grid_thw")

            if pixel_values is None:
                 if return_spatial_grid:
                    return torch.zeros(_QWEN2_VIT_DIM, 14, 14)
                 return torch.zeros(_QWEN2_VIT_DIM)

            with torch.no_grad():
                # Extract from the ViT encoder (visual tower) before the language head.
                # Output shape: (total_visual_tokens, hidden_size)
                visual_features = self.model.model.visual(
                    pixel_values,
                    grid_thw=image_grid_thw,
                )
                
                if return_spatial_grid and image_grid_thw is not None:
                    # Qwen2-VL image_grid_thw format: [T, H, W] per image. 
                    # Assuming single image, so T=1. We want features arranged as [C, H, W]
                    thw = image_grid_thw[0] if image_grid_thw.dim() > 1 else image_grid_thw
                    t, h, w = thw[0].item(), thw[1].item(), thw[2].item()
                    
                    # Each token in Qwen2-VL's ViT is actually a 2x2 patch on the original grid if using 
                    # their specific patch merge architecture, but the final sequence length matches h*w.
                    
                    # Reshape to [H, W, C] then permute to [C, H, W]
                    # We take the first h*w tokens in case there are multiple images, though here there's just 1
                    tokens = visual_features[:h*w]
                    grid = tokens.view(h, w, _QWEN2_VIT_DIM).permute(2, 0, 1)
                    return grid.cpu().float()
                else:
                    # Mean pool across all visual tokens to get a fixed-size embedding.
                    embedding = visual_features.mean(dim=0)
                    return embedding.cpu().float()

        except Exception as e:
            logger.warning(f"Visual embedding extraction failed: {e}")
            if return_spatial_grid:
                return torch.zeros(_QWEN2_VIT_DIM, 14, 14)
            return torch.zeros(_QWEN2_VIT_DIM)

    # Alias kept for backward compatibility with code calling get_embeddings().
    def get_embeddings(self, image_input: Any) -> torch.Tensor:
        return self.get_visual_embeddings(image_input)

    def process_stream_frame(self, frame: Any) -> dict[str, Any]:
        """
        Process a single frame from a real-time stream.
        Returns embedding dict compatible with MultimodalEmotionDetector.
        """
        embedding = self.get_visual_embeddings(frame)
        return {"embedding": embedding, "description": ""}