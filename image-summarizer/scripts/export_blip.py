"""
export_blip.py — Export BLIP image captioning model for on-device deployment.

BLIP has two components:
    1. Vision Encoder (ViT): Can be compiled for NPU (single forward pass)
    2. Text Decoder: Runs on CPU (autoregressive, generates one token at a time)

This script exports the vision encoder to ONNX and optionally compiles it
via Qualcomm AI Hub Workbench for NPU acceleration.

Usage:
    python scripts/export_blip.py                   # Local ONNX export
    python scripts/export_blip.py --compile          # Compile for NPU via AI Hub
    python scripts/export_blip.py --variant large    # Use larger model
"""

import os
import sys
import argparse

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def parse_args():
    parser = argparse.ArgumentParser(description="Export BLIP for on-device deployment")
    parser.add_argument(
        "--variant",
        choices=["base", "large"],
        default="base",
        help="Model variant: base (faster) or large (better quality)",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile via Qualcomm AI Hub for NPU acceleration",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="Snapdragon X Elite CRD",
        help="Target device for AI Hub compilation",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(project_root, "models"),
        help="Output directory",
    )
    return parser.parse_args()


def export_vision_encoder(args):
    """Export BLIP's vision encoder to ONNX format."""
    import torch
    from transformers import BlipForConditionalGeneration

    model_name = (
        "Salesforce/blip-image-captioning-base"
        if args.variant == "base"
        else "Salesforce/blip-image-captioning-large"
    )

    print(f"[BLIP Export] Loading model: {model_name}")
    model = BlipForConditionalGeneration.from_pretrained(model_name)
    model.eval()

    # Extract just the vision encoder
    vision_model = model.vision_model

    # Input dimensions
    if args.variant == "base":
        input_size = 384  # BLIP-base uses 384x384
    else:
        input_size = 384  # BLIP-large also uses 384x384

    dummy_input = torch.randn(1, 3, input_size, input_size)

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "blip_vision_encoder.onnx")

    print(f"[BLIP Export] Exporting vision encoder to ONNX...")
    print(f"[BLIP Export] Input size: 1x3x{input_size}x{input_size}")

    torch.onnx.export(
        vision_model,
        dummy_input,
        output_path,
        input_names=["pixel_values"],
        output_names=["last_hidden_state", "pooler_output"],
        dynamic_axes={
            "pixel_values": {0: "batch_size"},
            "last_hidden_state": {0: "batch_size"},
            "pooler_output": {0: "batch_size"},
        },
        opset_version=14,
        do_constant_folding=True,
    )

    file_size = os.path.getsize(output_path) / 1024 / 1024
    print(f"[BLIP Export] Saved to: {output_path}")
    print(f"[BLIP Export] File size: {file_size:.1f} MB")

    return output_path


def compile_for_npu(onnx_path, args):
    """Compile the ONNX model via Qualcomm AI Hub for NPU acceleration."""
    import qai_hub as hub

    print(f"[BLIP Export] Compiling for device: {args.device}")

    input_size = 384
    compile_job = hub.submit_compile_job(
        model=onnx_path,
        device=hub.Device(args.device),
        options="--target_runtime onnx",
        input_specs={"pixel_values": (1, 3, input_size, input_size)},
    )

    print(f"[BLIP Export] Compile job: {compile_job}")
    print("[BLIP Export] Waiting for compilation...")

    target_model = compile_job.get_target_model()
    output_path = os.path.join(args.output_dir, "blip_vision_optimized.onnx")
    target_model.download(output_path)

    print(f"[BLIP Export] Optimized model saved to: {output_path}")
    return output_path


def main():
    args = parse_args()

    print("=" * 60)
    print("  BLIP Model Export")
    print("=" * 60)

    # Step 1: Export to ONNX
    onnx_path = export_vision_encoder(args)

    # Step 2: Optionally compile for NPU
    if args.compile:
        try:
            compile_for_npu(onnx_path, args)
        except ImportError:
            print("[BLIP Export] qai-hub not installed. Skipping NPU compilation.")
            print("[BLIP Export] Install with: pip install qai-hub")
    else:
        print()
        print("[BLIP Export] To compile for NPU, run:")
        print(f"  python scripts/export_blip.py --compile --device \"{args.device}\"")

    print("\n[BLIP Export] Done!")


if __name__ == "__main__":
    main()
