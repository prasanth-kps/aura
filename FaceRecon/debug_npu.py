import onnxruntime as ort
import glob, os

# Find all ONNX files that were downloaded
onnx_files = glob.glob('./whisper_onnx_hf/**/*.onnx', recursive=True)
print("Found ONNX files:")
for f in onnx_files:
    size = os.path.getsize(f) / (1024*1024)
    print(f"  {f}  ({size:.1f} MB)")

QNN_OPTIONS = {
    "backend_path": r"C:\Windows\WinSxS\arm64_winappsdk-cbs_31bf3856ad364e35_10.0.26100.1_none_c1de85a65c8cf49a\QnnHtp.dll",
    "htp_performance_mode": "burst",
}

# Try loading encoder
encoder_path = './whisper_onnx_hf/onnx/encoder_model.onnx'
if os.path.exists(encoder_path):
    print("\nLoading encoder...")
    enc = ort.InferenceSession(
        encoder_path,
        providers=[("QNNExecutionProvider", QNN_OPTIONS), "CPUExecutionProvider"]
    )
    print("SUCCESS! Provider:", enc.get_providers())
    print("Inputs:", [i.name for i in enc.get_inputs()])