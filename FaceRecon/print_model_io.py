"""
print_model_io.py
Reads IO spec directly from the ONNX protobuf — works even for EPContext models
where get_inputs()/get_outputs() returns empty.
"""
import onnx
from pathlib import Path

MODEL_DIR = Path("whisper_npu_model")

for name in ("whisper_encoder.onnx", "whisper_decoder.onnx"):
    path = MODEL_DIR / name
    print(f"\n{'='*65}")
    print(f"  {name}")
    print('='*65)
    m = onnx.load(str(path), load_external_data=False)

    print("  INPUTS:")
    for inp in m.graph.input:
        shape = [d.dim_value if d.dim_value > 0 else d.dim_param
                 for d in inp.type.tensor_type.shape.dim] if inp.type.tensor_type.HasField('shape') else "?"
        dtype = inp.type.tensor_type.elem_type
        print(f"    {inp.name:<48} shape={str(shape):<30} dtype={dtype}")

    print("  OUTPUTS:")
    for out in m.graph.output:
        shape = [d.dim_value if d.dim_value > 0 else d.dim_param
                 for d in out.type.tensor_type.shape.dim] if out.type.tensor_type.HasField('shape') else "?"
        dtype = out.type.tensor_type.elem_type
        print(f"    {out.name:<48} shape={str(shape):<30} dtype={dtype}")

    print("\n  EPContext node attributes:")
    for node in m.graph.node:
        if node.op_type == "EPContext":
            for attr in node.attribute:
                if attr.name != "ep_cache_context":  # skip binary blob
                    val = attr.s.decode() if attr.s else attr.i or attr.f
                    print(f"    {attr.name} = {val}")