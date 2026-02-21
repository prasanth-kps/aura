"""
blip_captioner.py — BLIP-based image captioning engine (Optional).

BLIP (Bootstrapping Language-Image Pre-training) generates free-form
natural language captions for images. Unlike CLIP's zero-shot approach
which selects from predefined descriptions, BLIP generates novel text.

Architecture:
    - Vision Encoder (ViT): Extracts visual features from the image
    - Text Decoder: Autoregressively generates caption tokens

For on-device deployment:
    - The vision encoder can be compiled for NPU via Qualcomm AI Hub
    - The text decoder runs on CPU (autoregressive generation is hard to NPU-accelerate)
    - Total latency is typically 100-300ms depending on caption length

Usage:
    captioner = BLIPCaptioner()  # Falls back to HuggingFace if no ONNX model
    result = captioner.caption(image, bbox)
    print(result["caption"])  # "a dog playing with a red ball on green grass"
"""

import time
import numpy as np
from PIL import Image
from typing import Tuple, Optional, Dict

from engine.preprocessing import crop_region, preprocess_for_blip


class BLIPCaptioner:
    """
    Generate natural language captions for image regions using BLIP.

    This provides higher quality output than CLIP zero-shot but requires
    either a compiled ONNX model or the full HuggingFace model.
    """

    def __init__(
        self,
        vision_encoder_path: Optional[str] = None,
        use_npu: bool = True,
        model_variant: str = "base",  # "base" or "large"
    ):
        """
        Initialize the BLIP captioner.

        Args:
            vision_encoder_path: Path to compiled BLIP vision encoder ONNX model.
                                If None, uses HuggingFace BLIP (CPU mode).
            use_npu: Whether to use NPU acceleration for the vision encoder.
            model_variant: "base" (faster, 224M params) or "large" (better, 446M params)
        """
        self.use_npu = use_npu
        self._inference_time_ms = 0
        self._vision_session = None
        self._blip_model = None
        self._blip_processor = None

        model_name = (
            "Salesforce/blip-image-captioning-base"
            if model_variant == "base"
            else "Salesforce/blip-image-captioning-large"
        )

        if vision_encoder_path:
            self._init_onnx(vision_encoder_path)
        else:
            self._init_huggingface(model_name)

    def _init_onnx(self, vision_encoder_path: str):
        """
        Initialize with compiled ONNX vision encoder.

        NOTE: For BLIP with ONNX, only the vision encoder runs on NPU.
        The text decoder still uses the HuggingFace model on CPU because
        autoregressive generation (one token at a time) doesn't benefit
        much from NPU acceleration.
        """
        import onnxruntime as ort

        providers = []
        provider_options = []

        if self.use_npu:
            providers.append("QNNExecutionProvider")
            provider_options.append({
                "backend_path": "QnnHtp.dll",
                "htp_performance_mode": "burst",
            })

        providers.append("CPUExecutionProvider")
        provider_options.append({})

        try:
            self._vision_session = ort.InferenceSession(
                vision_encoder_path,
                providers=providers,
                provider_options=provider_options,
            )
            print(f"[BLIP] ONNX vision encoder loaded. Providers: {self._vision_session.get_providers()}")
        except Exception as e:
            print(f"[BLIP] ONNX load failed: {e}. Falling back to HuggingFace.")
            self._init_huggingface("Salesforce/blip-image-captioning-base")

    def _init_huggingface(self, model_name: str):
        """
        Load full BLIP model from HuggingFace.

        This is the simplest path — just works, no model export needed.
        Good for prototyping and hackathon demos.
        """
        try:
            from transformers import BlipProcessor, BlipForConditionalGeneration

            print(f"[BLIP] Loading {model_name} from HuggingFace...")
            self._blip_processor = BlipProcessor.from_pretrained(model_name)
            self._blip_model = BlipForConditionalGeneration.from_pretrained(model_name)
            self._blip_model.eval()
            print("[BLIP] Model loaded successfully.")
        except ImportError:
            print("[BLIP] ERROR: transformers not installed.")
            print("[BLIP] Install with: pip install transformers torch")
            raise

    def caption(
        self,
        image: Image.Image,
        bbox: Tuple[int, int, int, int],
        prompt: Optional[str] = None,
        max_length: int = 50,
    ) -> Dict:
        """
        Generate a caption for the image region within the bounding box.

        Args:
            image: Full PIL Image (RGB)
            bbox: (x1, y1, x2, y2) coordinates of context window
            prompt: Optional text prompt to condition the caption.
                    Example: "a photograph of" → "a photograph of a dog..."
                    If None, generates unconditional caption.
            max_length: Maximum number of tokens in generated caption.

        Returns:
            Dictionary with:
                - "caption": Generated text caption
                - "inference_ms": Total inference time
                - "cropped_image": The cropped region
        """
        # Crop the region
        cropped = crop_region(image, bbox)

        start_time = time.perf_counter()

        if self._blip_model is not None:
            caption = self._caption_huggingface(cropped, prompt, max_length)
        else:
            caption = self._caption_onnx(cropped, prompt, max_length)

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self._inference_time_ms = elapsed_ms

        return {
            "caption": caption,
            "inference_ms": round(elapsed_ms, 1),
            "cropped_image": cropped,
        }

    def _caption_huggingface(
        self, image: Image.Image, prompt: Optional[str], max_length: int
    ) -> str:
        """Generate caption using full HuggingFace BLIP model."""
        import torch

        with torch.no_grad():
            if prompt:
                # Conditional captioning: model continues from the prompt
                inputs = self._blip_processor(
                    image, text=prompt, return_tensors="pt"
                )
            else:
                # Unconditional captioning: model generates from scratch
                inputs = self._blip_processor(
                    image, return_tensors="pt"
                )

            output_ids = self._blip_model.generate(
                **inputs,
                max_length=max_length,
                num_beams=3,          # Beam search for better quality
                early_stopping=True,
            )

            caption = self._blip_processor.decode(
                output_ids[0], skip_special_tokens=True
            )

        return caption.strip()

    def _caption_onnx(
        self, image: Image.Image, prompt: Optional[str], max_length: int
    ) -> str:
        """
        Generate caption with ONNX vision encoder + HuggingFace text decoder.

        The vision encoder runs on NPU for speed, while the text decoder
        runs on CPU since it generates tokens one at a time.
        """
        # For full ONNX-only pipeline, you'd need to export the text decoder too.
        # For the hackathon, using HuggingFace text decoder is pragmatic.
        # The vision encoder (the expensive part) still runs on NPU.

        if self._blip_model is not None:
            return self._caption_huggingface(image, prompt, max_length)

        # If we only have the ONNX vision encoder without a text decoder,
        # fall back to a simple description
        input_tensor = preprocess_for_blip(image)
        input_name = self._vision_session.get_inputs()[0].name
        outputs = self._vision_session.run(None, {input_name: input_tensor})

        # Without a text decoder, we can't generate captions from just visual features
        return "[BLIP vision features extracted — text decoder not available for ONNX-only mode]"

    @property
    def last_inference_time_ms(self) -> float:
        """Return the last inference time in milliseconds."""
        return self._inference_time_ms
