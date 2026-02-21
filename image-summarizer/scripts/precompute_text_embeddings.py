"""
precompute_text_embeddings.py — Pre-compute CLIP text embeddings for all prompts.

This script runs the CLIP text encoder once on all description prompts
and saves the resulting embeddings as a NumPy .npy file. At runtime,
only the image encoder needs to run (on NPU), while text embeddings
are loaded directly from disk.

This is a critical optimization:
    - Text encoding: ~50ms per batch on CPU
    - Loading pre-computed embeddings: <1ms
    - Since prompts don't change at runtime, there's no reason to
      re-encode them every time the app starts.

Usage:
    python scripts/precompute_text_embeddings.py
    python scripts/precompute_text_embeddings.py --prompts models/prompts.json
    python scripts/precompute_text_embeddings.py --output models/clip_text_embeddings.npy
"""

import os
import sys
import json
import argparse
import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pre-compute CLIP text embeddings for all description prompts"
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default=os.path.join(project_root, "models", "prompts.json"),
        help="Path to prompts.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(project_root, "models", "clip_text_embeddings.npy"),
        help="Output path for .npy file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/clip-vit-base-patch16",
        help="HuggingFace CLIP model name",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  Pre-compute CLIP Text Embeddings")
    print("=" * 60)

    # Load prompts
    print(f"[TextEmb] Loading prompts from: {args.prompts}")
    with open(args.prompts, "r") as f:
        prompts_data = json.load(f)

    all_prompts = (
        prompts_data["scene_descriptions"]
        + prompts_data["attribute_descriptions"]
        + prompts_data["action_descriptions"]
    )
    print(f"[TextEmb] Total prompts: {len(all_prompts)}")

    # Load CLIP text encoder
    print(f"[TextEmb] Loading CLIP model: {args.model}")
    import torch
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(args.model)
    processor = CLIPProcessor.from_pretrained(args.model)
    model.eval()

    # Encode all prompts
    print("[TextEmb] Encoding text prompts...")
    with torch.no_grad():
        inputs = processor(
            text=all_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        text_features = model.get_text_features(**inputs)

        # L2 normalize (required for cosine similarity via dot product)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # Convert to numpy and save
    embeddings = text_features.numpy()
    print(f"[TextEmb] Embedding shape: {embeddings.shape}")
    print(f"[TextEmb] Embedding dtype: {embeddings.dtype}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.save(args.output, embeddings)
    print(f"[TextEmb] Saved to: {args.output}")
    print(f"[TextEmb] File size: {os.path.getsize(args.output) / 1024:.1f} KB")

    # Verify by loading back
    loaded = np.load(args.output)
    assert loaded.shape == embeddings.shape, "Shape mismatch!"
    assert np.allclose(loaded, embeddings, atol=1e-6), "Value mismatch!"
    print("[TextEmb] Verification passed.")

    # Print prompt-to-index mapping for reference
    print("\n[TextEmb] Prompt index mapping:")
    for i, prompt in enumerate(all_prompts):
        print(f"  [{i:3d}] {prompt}")

    print(f"\n[TextEmb] Done! Use this file with CLIPSummarizer:")
    print(f'  CLIPSummarizer(text_embeddings_path="{args.output}")')


if __name__ == "__main__":
    main()
