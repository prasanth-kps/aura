"""
fix_whisper_npu.py  (v2)
========================
The .bin files exist but are named whisper_encoder.bin / whisper_decoder.bin
while the .onnx shells expect ./model.bin

Fix: patch each .onnx to point to its own .bin file by name.
Then smoke-test on NPU.

Usage:
    py fix_whisper_npu.py
"""

from pathlib import Path
import sys

MODEL_DIR = Path("whisper_npu_model")

# ── Step 1: patch .onnx ep_cache_context to correct .bin name ────────────────

def patch_onnx():
    try:
        import onnx
    except ImportError:
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "onnx", "--user", "-q"])
        import onnx

    mapping = {
        "whisper_encoder.onnx": "whisper_encoder.bin",
        "whisper_decoder.onnx": "whisper_decoder.bin",
    }

    print("Patching .onnx files...")
    for onnx_name, bin_name in mapping.items():
        onnx_path = MODEL_DIR / onnx_name
        bin_path  = MODEL_DIR / bin_name

        if not onnx_path.exists():
            print(f"  SKIP: {onnx_name} not found")
            continue
        if not bin_path.exists():
            print(f"  SKIP: {bin_name} not found")
            continue

        m = onnx.load(str(onnx_path), load_external_data=False)
        patched = False
        for node in m.graph.node:
            for attr in node.attribute:
                if attr.name == "ep_cache_context":
                    old = attr.s.decode()
                    attr.s = f"./{bin_name}".encode()
                    print(f"  {onnx_name}: '{old}' → './{bin_name}'")
                    patched = True

        if patched:
            onnx.save(m, str(onnx_path))
            print(f"  Saved: {onnx_name} ✓")
        else:
            print(f"  {onnx_name}: no ep_cache_context attribute found")


# ── Step 2: smoke test both sessions on NPU ──────────────────────────────────

def smoke_test():
    import onnxruntime as ort
    import numpy as np

    provider_opts = [{"backend_path": "QnnHtp.dll",
                      "htp_performance_mode": "burst",
                      "htp_graph_finalization_optimization_mode": "3"}]

    print("\n" + "=" * 60)
    for name in ("whisper_encoder.onnx", "whisper_decoder.onnx"):
        path = MODEL_DIR / name
        print(f"\nLoading {name} on NPU...")
        try:
            sess = ort.InferenceSession(
                str(path),
                providers=["QNNExecutionProvider"],
                provider_options=provider_opts,
            )
            print(f"  INPUTS:")
            for i in sess.get_inputs():
                print(f"    {i.name:<45} shape={str(i.shape):<25} dtype={i.type}")
            print(f"  OUTPUTS:")
            for o in sess.get_outputs():
                print(f"    {o.name:<45} shape={str(o.shape):<25} dtype={o.type}")
            print(f"  ✓ {name} loaded successfully!")
        except Exception as e:
            print(f"  ✗ Failed: {e}")


if __name__ == "__main__":
    patch_onnx()
    smoke_test()