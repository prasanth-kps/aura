"""
clip_summarizer.py — CLIP-based zero-shot image region summarizer.

This is the primary summarization engine for the app. It uses OpenAI's CLIP
model (available on Qualcomm AI Hub) to understand what's in a cropped image
region by comparing the image against a bank of text descriptions.

How CLIP zero-shot classification works:
    1. CLIP has two encoders: one for images, one for text
    2. Both encoders project their input into the same embedding space
    3. Similar images and text descriptions end up close together
    4. We pre-encode all our text descriptions once at startup
    5. At runtime, we only encode the image (fast!) and find the closest texts
    6. The top matching descriptions become the summary

This approach is ideal for a hackathon because:
    - CLIP is already optimized on Qualcomm AI Hub (no manual export needed)
    - Image encoding is a single forward pass (~20-50ms on NPU)
    - Text embeddings are pre-computed (zero runtime cost)
    - The prompt bank can be easily expanded for better descriptions
"""

import json
import os
import time
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional, Dict
from pathlib import Path

from engine.preprocessing import preprocess_for_clip, crop_region


class CLIPSummarizer:
    """
    Summarizes image regions using CLIP zero-shot classification.

    The summarizer maintains:
        - An ONNX Runtime session for the CLIP image encoder (runs on NPU)
        - Pre-computed text embeddings for all description prompts
        - Template strings for composing natural language summaries
    """

    def __init__(
        self,
        image_encoder_path: Optional[str] = None,
        text_embeddings_path: Optional[str] = None,
        prompts_path: Optional[str] = None,
        use_npu: bool = True,
    ):
        """
        Initialize the CLIP summarizer.

        Args:
            image_encoder_path: Path to compiled CLIP image encoder ONNX model.
                               If None, falls back to HuggingFace CLIP in PyTorch.
            text_embeddings_path: Path to .npy file of pre-computed text embeddings.
                                  If None, computes them at startup (slower first run).
            prompts_path: Path to prompts.json with description text bank.
            use_npu: If True, attempt to use QNN Execution Provider for NPU acceleration.
        """
        self.use_npu = use_npu
        self._inference_time_ms = 0

        # Load prompts
        if prompts_path is None:
            prompts_path = os.path.join(
                os.path.dirname(__file__), "..", "models", "prompts.json"
            )
        self.prompts_data = self._load_prompts(prompts_path)

        # Build flat list of all description prompts
        self.all_prompts = (
            self.prompts_data["scene_descriptions"]
            + self.prompts_data["attribute_descriptions"]
            + self.prompts_data["action_descriptions"]
        )
        self.num_scene = len(self.prompts_data["scene_descriptions"])
        self.num_attr = len(self.prompts_data["attribute_descriptions"])
        self.num_action = len(self.prompts_data["action_descriptions"])

        # Initialize the model
        self._session = None
        self._text_embeddings = None
        self._clip_model = None
        self._clip_processor = None

        if image_encoder_path and os.path.exists(image_encoder_path):
            self._init_onnx_session(image_encoder_path)
            if text_embeddings_path and os.path.exists(text_embeddings_path):
                self._text_embeddings = np.load(text_embeddings_path)
            else:
                print("[CLIP] No pre-computed text embeddings found.")
                print("[CLIP] Run scripts/precompute_text_embeddings.py first.")
                print("[CLIP] Falling back to HuggingFace CLIP for text encoding.")
                self._init_huggingface_text_encoder()
        else:
            print("[CLIP] No compiled ONNX model found at:", image_encoder_path)
            print("[CLIP] Falling back to HuggingFace CLIP (CPU mode).")
            print("[CLIP] For NPU acceleration, export the model using:")
            print("[CLIP]   python scripts/export_clip.py")
            self._init_huggingface_full()

    # ================================================================
    # Initialization Methods
    # ================================================================

    def _load_prompts(self, path: str) -> Dict:
        """Load the description prompt bank from JSON."""
        try:
            with open(path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"[CLIP] Prompts file not found at {path}, using defaults.")
            return self._default_prompts()

    def _default_prompts(self) -> Dict:
        """Fallback prompts if prompts.json is missing."""
        return {
            "scene_descriptions": [
                "a photograph of a person",
                "a photograph of an animal",
                "a photograph of a building",
                "a photograph of nature or landscape",
                "a photograph of food",
                "a photograph of a vehicle",
                "a photograph of text or writing",
                "a photograph of an object",
            ],
            "attribute_descriptions": [
                "a bright and colorful image",
                "a dark image",
                "an indoor scene",
                "an outdoor scene",
            ],
            "action_descriptions": [
                "someone doing an activity",
                "a still or static scene",
            ],
            "summary_templates": [
                "This region shows {scenes}. The image appears to be {attributes}."
            ],
        }

    def _init_onnx_session(self, model_path: str):
        """
        Initialize ONNX Runtime session with NPU acceleration.

        The provider fallback chain is:
            1. QNNExecutionProvider (Hexagon NPU — fastest)
            2. DmlExecutionProvider (GPU via DirectML — medium)
            3. CPUExecutionProvider (CPU — slowest but always works)
        """
        import onnxruntime as ort

        # Build provider list with fallbacks
        providers = []
        provider_options = []

        if self.use_npu:
            providers.append("QNNExecutionProvider")
            provider_options.append({
                "backend_path": "QnnHtp.dll",
                "htp_performance_mode": "burst",
            })

        # DirectML (GPU) fallback for Windows
        providers.append("DmlExecutionProvider")
        provider_options.append({})

        # CPU always available
        providers.append("CPUExecutionProvider")
        provider_options.append({})

        try:
            self._session = ort.InferenceSession(
                model_path,
                providers=providers,
                provider_options=provider_options,
            )
            active = self._session.get_providers()
            print(f"[CLIP] ONNX session created. Active providers: {active}")
        except Exception as e:
            print(f"[CLIP] ONNX session failed: {e}")
            print("[CLIP] Falling back to HuggingFace CLIP.")
            self._init_huggingface_full()

    def _init_huggingface_full(self):
        """
        Load full CLIP model from HuggingFace (both image + text encoder).
        This is the fallback when no compiled ONNX model is available.
        Runs on CPU/CUDA — no NPU acceleration.
        """
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor

            print("[CLIP] Loading HuggingFace CLIP model (this may take a moment)...")
            self._clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
            self._clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
            self._clip_model.eval()

            # Pre-compute text embeddings using the text encoder
            self._precompute_text_embeddings_hf()
            print("[CLIP] HuggingFace CLIP loaded successfully.")
        except ImportError:
            print("[CLIP] ERROR: transformers/torch not installed.")
            print("[CLIP] Install with: pip install transformers torch")
            raise

    def _init_huggingface_text_encoder(self):
        """Load only the text encoder from HuggingFace to compute text embeddings."""
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor

            self._clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
            self._clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
            self._clip_model.eval()
            self._precompute_text_embeddings_hf()
        except ImportError:
            print("[CLIP] ERROR: Cannot compute text embeddings without transformers/torch.")
            raise

    def _precompute_text_embeddings_hf(self):
        """
        Pre-compute text embeddings for all prompts using HuggingFace CLIP.

        This only runs once at startup. The embeddings are cached in memory
        and used for all subsequent similarity comparisons.
        """
        import torch

        with torch.no_grad():
            inputs = self._clip_processor(
                text=self.all_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            text_outputs = self._clip_model.get_text_features(**inputs)

            # L2 normalize for cosine similarity
            text_outputs = text_outputs / text_outputs.norm(dim=-1, keepdim=True)
            self._text_embeddings = text_outputs.numpy()

        print(f"[CLIP] Pre-computed {len(self.all_prompts)} text embeddings.")

    # ================================================================
    # Inference Methods
    # ================================================================

    def summarize(
        self,
        image: Image.Image,
        bbox: Tuple[int, int, int, int],
        top_k_scenes: int = 3,
        top_k_attributes: int = 2,
        top_k_actions: int = 1,
    ) -> Dict:
        """
        Summarize the content within a bounding box region of an image.

        Args:
            image: Full PIL Image (RGB)
            bbox: (x1, y1, x2, y2) coordinates of the context window
            top_k_scenes: Number of top scene descriptions to include
            top_k_attributes: Number of top attribute descriptions to include
            top_k_actions: Number of top action descriptions to include

        Returns:
            Dictionary with:
                - "summary": Human-readable text summary
                - "scenes": List of (description, score) tuples
                - "attributes": List of (description, score) tuples
                - "actions": List of (description, score) tuples
                - "inference_ms": Inference time in milliseconds
                - "cropped_image": The cropped PIL Image for display
        """
        # Step 1: Crop the region
        cropped = crop_region(image, bbox)

        # Step 2: Get image embedding
        start_time = time.perf_counter()
        image_embedding = self._encode_image(cropped)
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self._inference_time_ms = elapsed_ms

        # Step 3: Compute similarities with all text descriptions
        similarities = self._compute_similarities(image_embedding)

        # Step 4: Extract top matches from each category
        scene_sims = similarities[: self.num_scene]
        attr_sims = similarities[self.num_scene : self.num_scene + self.num_attr]
        action_sims = similarities[self.num_scene + self.num_attr :]

        scenes = self._top_k_with_labels(
            scene_sims, self.prompts_data["scene_descriptions"], top_k_scenes
        )
        attributes = self._top_k_with_labels(
            attr_sims, self.prompts_data["attribute_descriptions"], top_k_attributes
        )
        actions = self._top_k_with_labels(
            action_sims, self.prompts_data["action_descriptions"], top_k_actions
        )

        # Step 5: Compose natural language summary
        summary_text = self._compose_summary(scenes, attributes, actions)

        return {
            "summary": summary_text,
            "scenes": scenes,
            "attributes": attributes,
            "actions": actions,
            "inference_ms": round(elapsed_ms, 1),
            "cropped_image": cropped,
        }

    def _encode_image(self, image: Image.Image) -> np.ndarray:
        """
        Encode an image into CLIP's embedding space.

        If ONNX session is available (NPU path), uses that.
        Otherwise falls back to HuggingFace PyTorch model (CPU path).

        Returns:
            L2-normalized image embedding, shape (1, 512)
        """
        if self._session is not None:
            # === NPU Path (ONNX Runtime + QNN EP) ===
            input_tensor = preprocess_for_clip(image)

            input_name = self._session.get_inputs()[0].name
            outputs = self._session.run(None, {input_name: input_tensor})

            image_features = outputs[0]  # shape: (1, 512)
        else:
            # === CPU Fallback Path (HuggingFace) ===
            import torch

            with torch.no_grad():
                inputs = self._clip_processor(
                    images=image, return_tensors="pt"
                )
                image_features = self._clip_model.get_image_features(**inputs)
                image_features = image_features.numpy()

        # L2 normalize
        norm = np.linalg.norm(image_features, axis=-1, keepdims=True)
        image_features = image_features / (norm + 1e-8)

        return image_features

    def _compute_similarities(self, image_embedding: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between image embedding and all text embeddings.

        Since both are L2-normalized, cosine similarity = dot product.

        Args:
            image_embedding: shape (1, 512)

        Returns:
            Similarity scores array, shape (num_prompts,)
        """
        # dot product of normalized vectors = cosine similarity
        similarities = np.dot(image_embedding, self._text_embeddings.T)
        return similarities[0]  # Remove batch dimension

    def _top_k_with_labels(
        self, scores: np.ndarray, labels: List[str], k: int
    ) -> List[Tuple[str, float]]:
        """
        Get top-k scoring labels with their scores.

        Args:
            scores: Similarity scores for this category
            labels: Corresponding text descriptions
            k: Number of top results to return

        Returns:
            List of (label, score) tuples, sorted by score descending
        """
        k = min(k, len(scores))
        top_indices = np.argsort(scores)[::-1][:k]
        return [(labels[i], float(scores[i])) for i in top_indices]

    def _compose_summary(
        self,
        scenes: List[Tuple[str, float]],
        attributes: List[Tuple[str, float]],
        actions: List[Tuple[str, float]],
    ) -> str:
        """
        Compose a natural language summary from the top matching descriptions.

        The summary combines scene, attribute, and action descriptions into
        readable prose. Low-confidence matches (score < threshold) are excluded.
        """
        # Filter by confidence threshold
        confidence_threshold = 0.15
        confident_scenes = [s for s, score in scenes if score > confidence_threshold]
        confident_attrs = [a for a, score in attributes if score > confidence_threshold]
        confident_actions = [a for a, score in actions if score > confidence_threshold]

        parts = []

        # Scene description
        if confident_scenes:
            # Clean up the descriptions (remove "a photograph of" prefix)
            clean_scenes = []
            for s in confident_scenes:
                s = s.replace("a photograph of ", "").replace("a photo of ", "")
                clean_scenes.append(s)

            if len(clean_scenes) == 1:
                parts.append(f"This region shows {clean_scenes[0]}.")
            else:
                joined = ", ".join(clean_scenes[:-1]) + f" and {clean_scenes[-1]}"
                parts.append(f"This region shows {joined}.")
        else:
            parts.append("This region contains visual content.")

        # Attribute description
        if confident_attrs:
            clean_attrs = []
            for a in confident_attrs:
                a = a.replace("a ", "").replace("an ", "")
                clean_attrs.append(a)
            parts.append(f"The image appears to be {', '.join(clean_attrs)}.")

        # Action description
        if confident_actions:
            clean_actions = []
            for a in confident_actions:
                clean_actions.append(a)
            parts.append(f"It looks like {', '.join(clean_actions)}.")

        return " ".join(parts)

    @property
    def last_inference_time_ms(self) -> float:
        """Return the last inference time in milliseconds."""
        return self._inference_time_ms
