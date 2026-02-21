"""
preprocessing.py — Image preprocessing utilities for CLIP and BLIP models.

This module handles all image transformations needed before feeding images
to the AI models. Each model has specific requirements for input size,
normalization values, and tensor format.

Key concepts:
    - CLIP expects 224x224 images normalized with specific mean/std values
    - BLIP expects 384x384 images with different normalization
    - All models expect NCHW format (batch, channels, height, width)
    - Images must be RGB (3 channels), float32
"""

import numpy as np
from PIL import Image
from typing import Tuple, Optional


# ============================================================
# CLIP Preprocessing Constants
# These values come from OpenAI's CLIP training procedure.
# The model was trained with images normalized using these
# specific mean and standard deviation values per channel.
# ============================================================
CLIP_IMAGE_SIZE = 224
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

# ============================================================
# BLIP Preprocessing Constants
# BLIP uses ImageNet normalization and a larger input size.
# ============================================================
BLIP_IMAGE_SIZE = 384
BLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
BLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def crop_region(image: Image.Image, bbox: Tuple[int, int, int, int]) -> Image.Image:
    """
    Crop a rectangular region from an image.

    Args:
        image: Source PIL Image (RGB)
        bbox: Tuple of (x1, y1, x2, y2) — top-left and bottom-right corners
              in pixel coordinates.

    Returns:
        Cropped PIL Image (RGB)

    The bbox is clamped to image boundaries to prevent errors when the
    context window extends beyond the image edge.
    """
    width, height = image.size

    x1 = max(0, min(int(bbox[0]), width - 1))
    y1 = max(0, min(int(bbox[1]), height - 1))
    x2 = max(x1 + 1, min(int(bbox[2]), width))
    y2 = max(y1 + 1, min(int(bbox[3]), height))

    return image.crop((x1, y1, x2, y2))


def resize_with_padding(
    image: Image.Image,
    target_size: int,
    fill_color: Tuple[int, int, int] = (0, 0, 0)
) -> Image.Image:
    """
    Resize image to fit within target_size x target_size while maintaining
    aspect ratio, then pad the shorter dimension with fill_color.

    This prevents distortion that would occur from a naive resize of
    non-square images. CLIP and BLIP both expect square inputs.

    Args:
        image: Source PIL Image (RGB)
        target_size: Target dimension (both width and height)
        fill_color: RGB tuple for padding (default black)

    Returns:
        Square PIL Image of exactly target_size x target_size
    """
    w, h = image.size
    scale = target_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = image.resize((new_w, new_h), Image.BICUBIC)

    # Create square canvas and paste resized image centered
    padded = Image.new("RGB", (target_size, target_size), fill_color)
    paste_x = (target_size - new_w) // 2
    paste_y = (target_size - new_h) // 2
    padded.paste(resized, (paste_x, paste_y))

    return padded


def preprocess_for_clip(image: Image.Image) -> np.ndarray:
    """
    Full preprocessing pipeline for CLIP's image encoder.

    Steps:
        1. Resize to 224x224 (center crop to maintain aspect ratio)
        2. Convert to float32 in [0, 1] range
        3. Normalize with CLIP mean and std
        4. Transpose from HWC to CHW format
        5. Add batch dimension → NCHW

    Args:
        image: PIL Image (any size, RGB)

    Returns:
        numpy array of shape (1, 3, 224, 224), dtype float32
    """
    # Step 1: Resize — use center crop for best results
    image = _center_crop_resize(image, CLIP_IMAGE_SIZE)

    # Step 2: Convert to float array [0, 1]
    pixel_values = np.array(image, dtype=np.float32) / 255.0

    # Step 3: Normalize per-channel
    # This shifts the pixel distribution to match what CLIP saw during training
    pixel_values = (pixel_values - CLIP_MEAN) / CLIP_STD

    # Step 4: HWC → CHW (height, width, channels → channels, height, width)
    # Neural networks expect channels-first format
    pixel_values = pixel_values.transpose(2, 0, 1)

    # Step 5: Add batch dimension → (1, 3, 224, 224)
    pixel_values = np.expand_dims(pixel_values, axis=0)

    return pixel_values.astype(np.float32)


def preprocess_for_blip(image: Image.Image) -> np.ndarray:
    """
    Full preprocessing pipeline for BLIP's vision encoder.

    Same steps as CLIP but with a larger input size (384x384).

    Args:
        image: PIL Image (any size, RGB)

    Returns:
        numpy array of shape (1, 3, 384, 384), dtype float32
    """
    image = _center_crop_resize(image, BLIP_IMAGE_SIZE)

    pixel_values = np.array(image, dtype=np.float32) / 255.0
    pixel_values = (pixel_values - BLIP_MEAN) / BLIP_STD
    pixel_values = pixel_values.transpose(2, 0, 1)
    pixel_values = np.expand_dims(pixel_values, axis=0)

    return pixel_values.astype(np.float32)


def _center_crop_resize(image: Image.Image, target_size: int) -> Image.Image:
    """
    Resize the shortest edge to target_size, then center-crop to a square.

    This is the standard preprocessing for CLIP/BLIP and preserves the most
    important content (center of the image) without distortion.

    Example:
        Input: 800x600 image, target_size=224
        Step 1: Resize to 299x224 (scale shortest edge to 224)
        Step 2: Crop center 224x224 from the 299x224 image
    """
    image = image.convert("RGB")
    w, h = image.size

    # Resize shortest side to target_size
    if w < h:
        new_w = target_size
        new_h = int(h * target_size / w)
    else:
        new_h = target_size
        new_w = int(w * target_size / h)

    image = image.resize((new_w, new_h), Image.BICUBIC)

    # Center crop
    left = (new_w - target_size) // 2
    top = (new_h - target_size) // 2
    image = image.crop((left, top, left + target_size, top + target_size))

    return image


def image_to_display_thumbnail(
    image: Image.Image,
    max_size: int = 150
) -> Image.Image:
    """
    Create a small thumbnail for display in the UI summary panel.

    Args:
        image: Source cropped region
        max_size: Maximum dimension (width or height)

    Returns:
        Resized PIL Image maintaining aspect ratio
    """
    image = image.copy()
    image.thumbnail((max_size, max_size), Image.BICUBIC)
    return image
