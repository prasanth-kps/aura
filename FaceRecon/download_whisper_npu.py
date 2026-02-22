"""
download_whisper_npu.py
=======================
Downloads pre-compiled Whisper-Small-Quantized ONNX assets for
Snapdragon X Elite directly from Qualcomm AI Hub — bypassing the
AIMET import that crashes on Windows.

Requirements:
    py -m pip install qai-hub requests --user

Usage:
    py download_whisper_npu.py
    (will prompt for your AI Hub API token if not configured)
"""

import json
import os
import sys
import zipfile
from pathlib import Path

OUTPUT_DIR = Path("whisper_npu_model")

# Model IDs for Whisper-Small-Quantized on Snapdragon X Elite
# These are the pre-compiled PRECOMPILED_QNN_ONNX assets from AI Hub
ENCODER_MODEL_ID = "mn5eq6ov5"   # WhisperSmallEncoderQuantizable - Snapdragon X Elite
DECODER_MODEL_ID = "m31oq4xo5"   # WhisperSmallDecoderQuantizable - Snapdragon X Elite


def get_api_token() -> str:
    """Read token from qai-hub config or prompt user."""
    config_paths = [
        Path.home() / ".qai_hub" / "client.ini",
        Path.home() / ".qai-hub" / "client.ini",
        Path(os.environ.get("APPDATA", "")) / "qai_hub" / "client.ini",
    ]
    for p in config_paths:
        if p.exists():
            for line in p.read_text().splitlines():
                if line.strip().startswith("api_token"):
                    token = line.split("=", 1)[-1].strip()
                    if token:
                        print(f"[INFO] Using API token from {p}")
                        return token
    # Try env var
    token = os.environ.get("QAI_HUB_API_TOKEN", "")
    if token:
        return token

    print("\nNo AI Hub API token found.")
    print("Get a free token at: https://aihub.qualcomm.com → Sign In → Settings → API Token")
    print("Then run:  py -m qai_hub configure --api_token YOUR_TOKEN\n")
    token = input("Or paste your token here: ").strip()
    return token


def download_model(api_token: str, model_id: str, out_path: Path) -> bool:
    """Download a compiled model from AI Hub by model ID."""
    try:
        import requests
    except ImportError:
        raise RuntimeError("requests required: py -m pip install requests --user")

    headers = {"Authorization": f"Bearer {api_token}"}
    base    = "https://api.aihub.qualcomm.com"

    # Get model info
    print(f"[INFO] Fetching model {model_id} ...")
    r = requests.get(f"{base}/v1/models/{model_id}", headers=headers, timeout=30)
    if r.status_code == 401:
        print("[ERROR] Invalid API token.")
        return False
    if r.status_code == 404:
        print(f"[WARN] Model {model_id} not found — may need to re-export.")
        return False
    r.raise_for_status()
    info = r.json()
    name = info.get("name", model_id)
    print(f"[INFO] Model: {name}")

    # Download
    dl_url = f"{base}/v1/models/{model_id}/download"
    print(f"[INFO] Downloading {name} → {out_path} ...")
    with requests.get(dl_url, headers=headers, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        with open(out_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 256):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded * 100 // total
                    sys.stdout.write(f"\r  {pct}% ({downloaded // 1024}KB / {total // 1024}KB)")
                    sys.stdout.flush()
    print(f"\n[INFO] Saved: {out_path}")
    return True


def try_qai_hub_direct() -> bool:
    """Try using qai_hub Python client directly (avoids qai_hub_models import)."""
    try:
        import qai_hub as hub
    except ImportError:
        return False

    OUTPUT_DIR.mkdir(exist_ok=True)
    print("[INFO] Using qai_hub client to fetch pre-compiled assets...")

    try:
        # Download pre-compiled encoder
        enc_model = hub.get_model(ENCODER_MODEL_ID)
        enc_path  = OUTPUT_DIR / "whisper_encoder.onnx"
        enc_model.download(str(enc_path))
        print(f"[INFO] Encoder downloaded → {enc_path}")

        dec_model = hub.get_model(DECODER_MODEL_ID)
        dec_path  = OUTPUT_DIR / "whisper_decoder.onnx"
        dec_model.download(str(dec_path))
        print(f"[INFO] Decoder downloaded → {dec_path}")
        return True
    except Exception as e:
        print(f"[WARN] qai_hub direct download failed: {e}")
        return False


def export_via_fetch_script() -> bool:
    """
    Call the fetch_static_assets function directly from the export module,
    bypassing the __init__.py import that triggers the AIMET check.
    """
    import importlib.util, sys

    # Find the export.py file path without importing the package
    try:
        import qai_hub_models
        pkg_root = Path(qai_hub_models.__file__).parent
    except ImportError:
        return False

    export_py = pkg_root / "models" / "whisper_small_quantized" / "export.py"
    if not export_py.exists():
        print(f"[WARN] {export_py} not found")
        return False

    print(f"[INFO] Loading export.py directly from: {export_py}")
    try:
        spec   = importlib.util.spec_from_file_location("wsq_export", export_py)
        module = importlib.util.module_from_spec(spec)

        # Stub out the broken import before loading
        # Insert a fake 'qai_hub_models.models.whisper_small_quantized' module
        import types
        fake_pkg = types.ModuleType("qai_hub_models.models.whisper_small_quantized")
        fake_pkg.MODEL_ID = "whisper_small_quantized"
        sys.modules["qai_hub_models.models.whisper_small_quantized"] = fake_pkg

        spec.loader.exec_module(module)

        if hasattr(module, "fetch_static_assets"):
            OUTPUT_DIR.mkdir(exist_ok=True)
            module.fetch_static_assets(str(OUTPUT_DIR))
            return True
        else:
            print("[WARN] fetch_static_assets not found in export.py")
            print(f"[INFO] Functions available: {[x for x in dir(module) if not x.startswith('_')]}")
    except Exception as e:
        print(f"[WARN] Direct load failed: {e}")
    return False


def main():
    print("=" * 60)
    print("Whisper-Small-Quantized NPU Asset Downloader")
    print("Target: Snapdragon X Elite (PRECOMPILED_QNN_ONNX)")
    print("=" * 60)

    OUTPUT_DIR.mkdir(exist_ok=True)

    # Method 1: Direct export.py load (bypasses broken __init__)
    print("\n[Method 1] Trying direct export.py fetch...")
    if export_via_fetch_script():
        print_success()
        return

    # Method 2: qai_hub Python client
    print("\n[Method 2] Trying qai_hub client...")
    if try_qai_hub_direct():
        print_success()
        return

    # Method 3: Raw REST API download
    print("\n[Method 3] Trying AI Hub REST API...")
    token = get_api_token()
    if not token:
        print("[ERROR] No API token provided.")
        print_manual_instructions()
        return

    enc_ok = download_model(token, ENCODER_MODEL_ID, OUTPUT_DIR / "whisper_encoder.onnx")
    dec_ok = download_model(token, DECODER_MODEL_ID, OUTPUT_DIR / "whisper_decoder.onnx")

    if enc_ok and dec_ok:
        print_success()
    else:
        print_manual_instructions()


def print_success():
    print("\n" + "=" * 60)
    print("SUCCESS! Models downloaded to:", OUTPUT_DIR.resolve())
    print("=" * 60)
    print("\nFiles:")
    for f in OUTPUT_DIR.glob("*.onnx"):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name}  ({size_mb:.1f} MB)")
    print("\nNext step:")
    print("  py npu_whisper_loader.py videoplayback.mp4")


def print_manual_instructions():
    print("\n" + "=" * 60)
    print("MANUAL DOWNLOAD INSTRUCTIONS")
    print("=" * 60)
    print("""
The automated download failed. Download manually:

1. Go to: https://aihub.qualcomm.com/models/whisper_small_quantized
2. Click 'Download Model'  
3. Select device: Snapdragon X Elite CRD
4. Select runtime: PRECOMPILED_QNN_ONNX
5. Download both Encoder and Decoder .onnx files
6. Place them in:  whisper_npu_model/
   - whisper_npu_model/whisper_encoder.onnx
   - whisper_npu_model/whisper_decoder.onnx
7. Run: py npu_whisper_loader.py videoplayback.mp4
""")


if __name__ == "__main__":
    main()