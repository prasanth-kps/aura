"""
inference_engine.py — Main orchestrator for on-device image summarization.

This module ties together all the AI models and provides a single interface
for the UI layer to call. It manages:

    1. Model initialization and fallback logic
    2. Routing between CLIP (fast) and BLIP (detailed) modes
    3. Combining outputs from multiple models into a unified summary
    4. Performance monitoring (latency tracking)

The engine supports three summarization modes:
    - "clip":  Fast zero-shot classification (recommended for hackathon)
    - "blip":  Natural language captioning (higher quality, slower)
    - "hybrid": CLIP + BLIP combined (best output, most latency)
"""

import os
import time
from PIL import Image
from typing import Dict, Tuple, Optional
from enum import Enum


class SummaryMode(Enum):
    CLIP = "clip"
    BLIP = "blip"
    HYBRID = "hybrid"


class InferenceEngine:
    """
    Main inference engine — the single entry point for the UI.

    Usage:
        engine = InferenceEngine(mode="clip")
        result = engine.summarize(image, bbox=(100, 100, 400, 400))
        print(result["summary"])
        print(f"Took {result['total_ms']}ms")
    """

    def __init__(
        self,
        mode: str = "clip",
        clip_model_path: Optional[str] = None,
        clip_text_embeddings_path: Optional[str] = None,
        blip_model_path: Optional[str] = None,
        prompts_path: Optional[str] = None,
        use_npu: bool = True,
    ):
        """
        Initialize the inference engine.

        Args:
            mode: One of "clip", "blip", or "hybrid"
            clip_model_path: Path to compiled CLIP image encoder (.onnx)
            clip_text_embeddings_path: Path to pre-computed text embeddings (.npy)
            blip_model_path: Path to compiled BLIP vision encoder (.onnx)
            prompts_path: Path to prompts.json
            use_npu: Whether to attempt NPU acceleration
        """
        self.mode = SummaryMode(mode)
        self.use_npu = use_npu
        self._clip = None
        self._blip = None

        # Resolve default paths relative to project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        models_dir = os.path.join(project_root, "models")

        if prompts_path is None:
            prompts_path = os.path.join(models_dir, "prompts.json")

        # Initialize requested model(s)
        if self.mode in (SummaryMode.CLIP, SummaryMode.HYBRID):
            self._init_clip(clip_model_path, clip_text_embeddings_path, prompts_path)

        if self.mode in (SummaryMode.BLIP, SummaryMode.HYBRID):
            self._init_blip(blip_model_path)

        self._ready = (self._clip is not None) or (self._blip is not None)
        if self._ready:
            print(f"[Engine] Ready. Mode: {self.mode.value}, NPU: {use_npu}")
        else:
            print("[Engine] WARNING: No models loaded. Summaries will be empty.")

    def _init_clip(self, model_path, text_emb_path, prompts_path):
        """Initialize the CLIP summarizer."""
        try:
            from engine.clip_summarizer import CLIPSummarizer

            self._clip = CLIPSummarizer(
                image_encoder_path=model_path,
                text_embeddings_path=text_emb_path,
                prompts_path=prompts_path,
                use_npu=self.use_npu,
            )
        except Exception as e:
            print(f"[Engine] Failed to initialize CLIP: {e}")
            self._clip = None

    def _init_blip(self, model_path):
        """Initialize the BLIP captioner."""
        try:
            from engine.blip_captioner import BLIPCaptioner

            self._blip = BLIPCaptioner(
                vision_encoder_path=model_path,
                use_npu=self.use_npu,
            )
        except Exception as e:
            print(f"[Engine] Failed to initialize BLIP: {e}")
            self._blip = None

    @property
    def is_ready(self) -> bool:
        """Check if at least one model is loaded and ready."""
        return self._ready

    def get_active_models(self) -> list:
        """Return list of active model names."""
        models = []
        if self._clip is not None:
            models.append("CLIP")
        if self._blip is not None:
            models.append("BLIP")
        return models

    def summarize(
        self,
        image: Image.Image,
        bbox: Tuple[int, int, int, int],
    ) -> Dict:
        """
        Generate a summary for the specified region of the image.

        This is the main entry point called by the UI whenever the user
        positions the context window and requests a summary.

        Args:
            image: Full PIL Image (RGB)
            bbox: (x1, y1, x2, y2) coordinates of context window

        Returns:
            Dictionary with:
                - "summary": Final human-readable summary text
                - "details": Detailed breakdown (scores, individual model outputs)
                - "total_ms": Total inference time in milliseconds
                - "model_used": Which model(s) produced this summary
                - "cropped_image": The cropped region for display
        """
        if not self._ready:
            return {
                "summary": "No AI model loaded. Please check setup instructions.",
                "details": {},
                "total_ms": 0,
                "model_used": "none",
                "cropped_image": None,
            }

        start_time = time.perf_counter()

        if self.mode == SummaryMode.CLIP:
            result = self._summarize_clip(image, bbox)
        elif self.mode == SummaryMode.BLIP:
            result = self._summarize_blip(image, bbox)
        else:  # HYBRID
            result = self._summarize_hybrid(image, bbox)

        total_ms = (time.perf_counter() - start_time) * 1000
        result["total_ms"] = round(total_ms, 1)

        return result

    def _summarize_clip(self, image: Image.Image, bbox: Tuple) -> Dict:
        """Run CLIP-only summarization."""
        clip_result = self._clip.summarize(image, bbox)

        return {
            "summary": clip_result["summary"],
            "details": {
                "scenes": clip_result["scenes"],
                "attributes": clip_result["attributes"],
                "actions": clip_result["actions"],
                "clip_ms": clip_result["inference_ms"],
            },
            "model_used": "CLIP (zero-shot)",
            "cropped_image": clip_result["cropped_image"],
        }

    def _summarize_blip(self, image: Image.Image, bbox: Tuple) -> Dict:
        """Run BLIP-only captioning."""
        blip_result = self._blip.caption(image, bbox)

        return {
            "summary": blip_result["caption"],
            "details": {
                "blip_ms": blip_result["inference_ms"],
            },
            "model_used": "BLIP (captioning)",
            "cropped_image": blip_result["cropped_image"],
        }

    def _summarize_hybrid(self, image: Image.Image, bbox: Tuple) -> Dict:
        """
        Run both CLIP and BLIP, combine outputs.

        The hybrid approach gives the richest summary:
            - BLIP provides the natural language caption
            - CLIP provides structured scene/attribute/action labels
            - Combined: caption + structured metadata
        """
        parts = []
        details = {}
        cropped = None

        # BLIP caption (natural language)
        if self._blip is not None:
            blip_result = self._blip.caption(image, bbox)
            parts.append(blip_result["caption"])
            details["blip_caption"] = blip_result["caption"]
            details["blip_ms"] = blip_result["inference_ms"]
            cropped = blip_result["cropped_image"]

        # CLIP analysis (structured)
        if self._clip is not None:
            clip_result = self._clip.summarize(image, bbox)
            details["scenes"] = clip_result["scenes"]
            details["attributes"] = clip_result["attributes"]
            details["actions"] = clip_result["actions"]
            details["clip_ms"] = clip_result["inference_ms"]
            if cropped is None:
                cropped = clip_result["cropped_image"]

            # Add CLIP's structured insight if BLIP caption exists
            if parts:
                top_scene = clip_result["scenes"][0][0] if clip_result["scenes"] else ""
                top_attr = clip_result["attributes"][0][0] if clip_result["attributes"] else ""
                if top_scene:
                    clean = top_scene.replace("a photograph of ", "")
                    parts.append(f"(Detected: {clean})")
            else:
                parts.append(clip_result["summary"])

        summary = " ".join(parts)

        models_used = []
        if self._blip is not None:
            models_used.append("BLIP")
        if self._clip is not None:
            models_used.append("CLIP")

        return {
            "summary": summary,
            "details": details,
            "model_used": " + ".join(models_used),
            "cropped_image": cropped,
        }

    def change_mode(self, new_mode: str):
        """
        Switch summarization mode at runtime.

        Initializes any models that aren't already loaded.
        """
        new_mode_enum = SummaryMode(new_mode)

        # Initialize models if needed
        if new_mode_enum in (SummaryMode.BLIP, SummaryMode.HYBRID) and self._blip is None:
            self._init_blip(None)

        if new_mode_enum in (SummaryMode.CLIP, SummaryMode.HYBRID) and self._clip is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            prompts_path = os.path.join(project_root, "models", "prompts.json")
            self._init_clip(None, None, prompts_path)

        self.mode = new_mode_enum
        self._ready = (self._clip is not None) or (self._blip is not None)
        print(f"[Engine] Mode changed to: {self.mode.value}")
