"""
DEPRECATED. VideoLLaMA3 is no longer the primary visual backbone.

Architectural decision (2026-02-18): Qwen2-VL is the primary visual backbone.
- M-ROPE natively handles temporal/3D video positional encoding for live sim frames.
- Qwen3-VL-Embedding provides the forward-compatible embedding extraction path.
- Native HuggingFace integration, documented 4-bit quantization.

Use models/vision-language/qwen2/qwen2_integration.py instead.
This stub is kept only for backward-compatible imports.
"""

import logging
from typing import Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)


class VideoLLaMA3Config:
    """Deprecated stub config."""
    model_path: str = ""
    model_variant: str = "default"
    max_buffer_size: int = 32
    device: str = "cpu"


class VideoLLaMA3Integration:
    """
    DEPRECATED stub. Routes to Qwen2-VL where possible.

    This class exists only so existing imports do not crash. All new code
    should use models/vision-language/qwen2/qwen2_integration.py directly.
    """

    def __init__(self, config=None):
        logger.warning(
            "VideoLLaMA3Integration is deprecated. Use Qwen2VLIntegration instead. "
            "See models/vision-language/qwen2/qwen2_integration.py"
        )
        self.config = config or {}
        self.current_variant = "default"
        self.frame_buffer = []

    async def initialize(self):
        """No-op stub."""
        pass

    async def process_input(self, visual_input=None, audio_input=None, **kwargs) -> Dict[str, Any]:
        """Returns an empty perception dict."""
        return {"visual_context": {}, "embedding": None, "description": ""}

    def process_stream_frame(self, frame: Any) -> Dict[str, Any]:
        """Returns a zeroed embedding dict."""
        import torch
        return {"embedding": torch.zeros(1536), "description": ""}

    def process_frames(self, frames, query=None) -> Dict[str, Any]:
        return {"response": "", "token_count": 0}

    def process_video(self, video_path: str, query: Optional[str] = None) -> Dict:
        return {"response": "", "token_count": 0}

    def set_model_variant(self, variant: str) -> None:
        if variant not in ("default", "abliterated", "streaming"):
            raise ValueError(f"Invalid variant. Choose from ['default', 'abliterated', 'streaming']")
        self.current_variant = variant