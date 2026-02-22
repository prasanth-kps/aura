"""
inspect_npu_model.py
====================
Prints the exact input/output names and shapes of the downloaded
Whisper NPU ONNX files so we can build the correct inference code.

Usage:
    py inspect_npu_model.py
"""
import sys
from pathlib import Path

try:
    import onnxruntime as ort
except ImportError:
    print("onnxruntime-qnn required: py -m pip install onnxruntime-qnn --user")
    sys.exit(1)

MODEL_DIR = Path("whisper_npu_model")

def inspect(path: Path):
    print(f"\n{'='*70}")
    print(f"  {path.name}  ({path.stat().st_size // 1024} KB)")
    print('='*70)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    print("\n  INPUTS:")
    for i in sess.get_inputs():
        print(f"    {i.name:<45} shape={str(i.shape):<25} dtype={i.type}")

    print("\n  OUTPUTS:")
    for o in sess.get_outputs():
        print(f"    {o.name:<45} shape={str(o.shape):<25} dtype={o.type}")

for onnx_file in sorted(MODEL_DIR.glob("*.onnx")):
    inspect(onnx_file)

print("\nDone. Copy this output and share it to get the correct inference code.")