import argparse
import json
from pathlib import Path
from typing import Any

import torch
import qai_hub as hub
from qai_hub_models.models.cavaface import Model as CavaFaceModel


def _trace_model(model: torch.nn.Module, shape: tuple[int, ...]) -> Any:
    example = torch.randn(*shape)
    model.eval()
    return torch.jit.trace(model, example)


def compile_embedding_model(device_name: str, runtime: str) -> str:
    model = CavaFaceModel.from_pretrained()
    input_specs = {"image": (1, 3, 112, 112)}
    traced = _trace_model(model, input_specs["image"])

    print("Submitting compile job for cavaface...")
    compile_job = hub.submit_compile_job(
        model=traced,
        device=hub.Device(device_name),
        input_specs=input_specs,
        options=f"--target_runtime {runtime}",
    )
    print(f"Submitted embedding compile job: {compile_job.job_id}")
    return compile_job.job_id


def try_compile_detector(device_name: str, runtime: str) -> str | None:
    try:
        # Optional import because some environments miss detector dependencies.
        from qai_hub_models.models.face_det_lite import Model as FaceDetLiteModel
    except Exception as exc:
        print(f"Skipping face_det_lite compile (optional dependency missing): {exc}")
        return None

    detector = FaceDetLiteModel.from_pretrained()
    input_specs = {"image": (1, 3, 320, 320)}
    traced = _trace_model(detector, input_specs["image"])

    print("Submitting compile job for face_det_lite...")
    compile_job = hub.submit_compile_job(
        model=traced,
        device=hub.Device(device_name),
        input_specs=input_specs,
        options=f"--target_runtime {runtime}",
    )
    print(f"Submitted detector compile job: {compile_job.job_id}")
    return compile_job.job_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile hackathon models for Qualcomm AI Hub.")
    parser.add_argument(
        "--device",
        default="Samsung Galaxy S24 (Family)",
        help="QAI Hub target device name.",
    )
    parser.add_argument(
        "--runtime",
        default="tflite",
        choices=["tflite", "qnn"],
        help="Target runtime for compilation.",
    )
    parser.add_argument(
        "--out",
        default="compiled_models.json",
        help="Output metadata JSON file.",
    )
    args = parser.parse_args()

    result: dict[str, Any] = {
        "device": args.device,
        "runtime": args.runtime,
    }
    result["embedding_compile_job_id"] = compile_embedding_model(args.device, args.runtime)
    result["detector_compile_job_id"] = try_compile_detector(args.device, args.runtime)

    out_path = Path(args.out)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote compile metadata to {out_path}")
    print("Use run_recognition.py with this metadata file.")


if __name__ == "__main__":
    main()
