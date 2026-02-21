"""
export_clip.py — Export CLIP image encoder for on-device deployment.

This script downloads the CLIP model from Qualcomm AI Hub Models,
compiles it for the target Snapdragon device, and saves the optimized
ONNX model for use with ONNX Runtime + QNN Execution Provider.

Prerequisites:
    pip install qai-hub qai-hub-models torch
    qai-hub configure --api_token YOUR_API_TOKEN

Usage:
    python scripts/export_clip.py
    python scripts/export_clip.py --device "Samsung Galaxy S25"
    python scripts/export_clip.py --runtime onnx --device "Snapdragon X Elite CRD"
    python scripts/export_clip.py --local-only    # Export to ONNX without AI Hub

The script supports two modes:
    1. Cloud compilation (default): Uses Qualcomm AI Hub Workbench to compile
       and profile the model on a real cloud-hosted device.
    2. Local export (--local-only): Exports to standard ONNX format that can
       run on CPU. Useful for development without AI Hub access.
"""

import os
import sys
import argparse

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def parse_args():
    parser = argparse.ArgumentParser(description="Export CLIP for on-device deployment")
    parser.add_argument(
        "--device",
        type=str,
        default="Snapdragon X Elite CRD",
        help="Target device name from Qualcomm AI Hub (default: Snapdragon X Elite CRD)",
    )
    parser.add_argument(
        "--runtime",
        choices=["onnx", "precompiled_qnn_onnx", "tflite", "qnn_dlc"],
        default="onnx",
        help="Target runtime format (default: onnx)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(project_root, "models"),
        help="Output directory for compiled model",
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Export to ONNX locally without using AI Hub Workbench",
    )
    parser.add_argument(
        "--skip-profile",
        action="store_true",
        help="Skip profiling step (faster but no performance data)",
    )
    return parser.parse_args()


def export_via_ai_hub(args):
    """
    Export CLIP using Qualcomm AI Hub Workbench (cloud compilation).

    Steps:
        1. Load CLIP model from qai_hub_models
        2. Trace the PyTorch model
        3. Submit compile job to AI Hub (targets specific Snapdragon device)
        4. Optionally profile the compiled model on cloud-hosted device
        5. Download the optimized model
    """
    import torch
    import qai_hub as hub
    from qai_hub_models.models.openai_clip import Model

    print("[Export] Loading CLIP model from Qualcomm AI Hub Models...")
    torch_model = Model.from_pretrained()

    print(f"[Export] Target device: {args.device}")
    print(f"[Export] Target runtime: {args.runtime}")

    # Get model specs
    input_spec = torch_model.get_input_spec()
    sample_inputs = torch_model.sample_inputs()

    print(f"[Export] Input spec: {input_spec}")

    # Trace the model (required for compilation)
    print("[Export] Tracing model...")
    traced_model = torch.jit.trace(
        torch_model,
        [torch.tensor(data[0]) for _, data in sample_inputs.items()],
    )

    # Submit compile job
    print("[Export] Submitting compile job to Qualcomm AI Hub...")
    print("[Export] (This uploads the model and compiles on cloud — may take a few minutes)")

    device = hub.Device(args.device)

    compile_options = f"--target_runtime {args.runtime}"
    compile_job = hub.submit_compile_job(
        model=traced_model,
        device=device,
        options=compile_options,
        input_specs=input_spec,
    )

    print(f"[Export] Compile job submitted: {compile_job}")
    print("[Export] Waiting for compilation to complete...")

    # The job runs asynchronously; this waits for completion
    target_model = compile_job.get_target_model()

    # Download compiled model
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "clip_optimized.onnx")
    target_model.download(output_path)
    print(f"[Export] Compiled model saved to: {output_path}")

    # Optional: Profile on device
    if not args.skip_profile:
        print("[Export] Submitting profile job...")
        profile_job = hub.submit_profile_job(
            model=target_model,
            device=device,
        )
        print(f"[Export] Profile results:")
        print(f"         {profile_job}")

    print("[Export] Done! Model is ready for on-device inference.")
    return output_path


def export_local_onnx(args):
    """
    Export CLIP to standard ONNX format (local, no AI Hub needed).

    This produces a regular ONNX model that can run on CPU via ONNX Runtime.
    It won't be optimized for NPU, but it's useful for:
        - Development and testing without AI Hub access
        - Running on non-Snapdragon hardware
        - Verifying the model works before cloud compilation
    """
    import torch
    from transformers import CLIPModel

    print("[Export] Loading CLIP model from HuggingFace...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
    model.eval()

    # Export the image encoder (vision model) to ONNX
    print("[Export] Exporting CLIP image encoder to ONNX...")
    vision_model = model.vision_model

    # Dummy input for tracing
    dummy_input = torch.randn(1, 3, 224, 224)

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "clip_image_encoder.onnx")

    torch.onnx.export(
        vision_model,
        dummy_input,
        output_path,
        input_names=["pixel_values"],
        output_names=["image_features"],
        dynamic_axes={
            "pixel_values": {0: "batch_size"},
            "image_features": {0: "batch_size"},
        },
        opset_version=14,
        do_constant_folding=True,
    )

    print(f"[Export] ONNX model saved to: {output_path}")
    print(f"[Export] Model size: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")
    print()
    print("[Export] NOTE: This is a standard ONNX model (CPU only).")
    print("[Export] For NPU acceleration, use: python scripts/export_clip.py")
    print("[Export] (requires qai-hub package and Qualcomm AI Hub account)")

    return output_path


def main():
    args = parse_args()

    print("=" * 60)
    print("  CLIP Model Export for Qualcomm AI Hub")
    print("=" * 60)

    if args.local_only:
        export_local_onnx(args)
    else:
        try:
            import qai_hub
            export_via_ai_hub(args)
        except ImportError:
            print("[Export] qai-hub package not installed.")
            print("[Export] Install with: pip install qai-hub qai-hub-models")
            print("[Export] Falling back to local ONNX export...")
            export_local_onnx(args)


if __name__ == "__main__":
    main()
