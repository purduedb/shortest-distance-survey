"""Export a single .jit.pt model to ONNX format.

Usage: python export_onnx_models.py --model_path <path_to_jit_model>
"""
import argparse
import os
import tempfile
import numpy as np  ## Having this unused import avoid the "KMP_DUPLICATE_LIB_OK=TRUE" error on macOS

import torch
import onnx
from onnx.external_data_helper import convert_model_to_external_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="path to .jit.pt")
    parser.add_argument("--output_dir", default=None, help="output directory, default: saved_onnx_models/")
    parser.add_argument("--name", default=None, help="output basename, default: --model_path's basename")
    parser.add_argument("--batch_size", type=int, default=1024, help="dummy batch size for tracing")
    args = parser.parse_args()

    # Default name/dir: mirror the .jit.pt's own name, one level up in saved_onnx_models/
    name = args.name or os.path.basename(args.model_path).removesuffix(".jit.pt")
    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        model_dir = os.path.dirname(os.path.abspath(args.model_path))
        output_dir = os.path.join(os.path.dirname(model_dir), "saved_onnx_models")

    os.makedirs(output_dir, exist_ok=True)
    final_onnx = os.path.join(output_dir, f"{name}.onnx")
    data_name = f"{name}.onnx.data"

    # Load the model and set to eval mode
    model = torch.jit.load(args.model_path, map_location="cpu")
    model.eval()

    # Dummy inputs only used to trace shapes
    dummy_src = torch.zeros(args.batch_size, dtype=torch.long)
    dummy_dst = torch.zeros(args.batch_size, dtype=torch.long)

    # Export to a scratch tmp path first
    with tempfile.TemporaryDirectory(dir=output_dir) as tmpdir:
        tmp_onnx = os.path.join(tmpdir, "tmp.onnx")
        torch.onnx.export(
            model, (dummy_src, dummy_dst), tmp_onnx,
            input_names=["src", "dst"], output_names=["dist"],
            dynamic_axes={"src": {0: "batch"}, "dst": {0: "batch"}, "dist": {0: "batch"}},
            opset_version=17,
            dynamo=False,
        )
        # Large models split their weights into a auxiliary file
        has_external_data = any(f != "tmp.onnx" for f in os.listdir(tmpdir))
        if has_external_data:
            model_proto = onnx.load(tmp_onnx, load_external_data=True)
            convert_model_to_external_data(model_proto, all_tensors_to_one_file=True,
                                            location=data_name, size_threshold=1024)
            onnx.save_model(model_proto, final_onnx)
        else:
            # Small models: no split, just move the .onnx file to final location
            os.replace(tmp_onnx, final_onnx)

    print(f"Saving ONNX model: {final_onnx}")


if __name__ == "__main__":
    main()
