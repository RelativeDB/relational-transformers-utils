"""Checkpoint quantizers producing formats the relational-transformers loader accepts.

FP8 checkpoints store matrix weights as native ``float8_e4m3fn``. Int8 checkpoints
store an integer weight plus one scale per output row under ``<name>.q_scale``. Int4
checkpoints pack two values per byte in groups of 32 and store fp16 ``(scale, minimum)``
pairs under ``<name>.q4_scale``. ``relational_transformers.checkpoints.load_state``
expands all three during loading.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from relational_transformers.checkpoints import resolve_checkpoint
from safetensors.torch import load_file, save_file

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = torch.finfo(FP8_DTYPE).max
Q4_GROUP = 32

FORMATS = ("fp8", "int8", "int4")


def _is_matrix_weight(name: str, tensor: torch.Tensor) -> bool:
    return tensor.is_floating_point() and tensor.ndim == 2 and name.endswith(".weight")


def quantize_tensor_fp8(name: str, tensor: torch.Tensor) -> torch.Tensor:
    """Convert matrix weights to native E4M3 FP8 and retain other tensors."""

    if not _is_matrix_weight(name, tensor):
        return tensor
    return tensor.float().clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)


def quantize_tensor_int8(name: str, tensor: torch.Tensor) -> dict[str, torch.Tensor]:
    """Quantize one matrix weight to int8 with one scale per output row."""

    if not _is_matrix_weight(name, tensor):
        return {name: tensor}
    values = tensor.float()
    scale = values.abs().amax(dim=1).clamp_min(1e-12) / 127.0
    quantized = torch.round(values / scale.unsqueeze(1)).clamp(-127, 127).to(torch.int8)
    return {name: quantized, f"{name}.q_scale": scale}


def quantize_tensor_int4(name: str, tensor: torch.Tensor) -> dict[str, torch.Tensor]:
    """Pack one matrix weight into nibbles with fp16 scale/minimum pairs per group of 32.

    Weights whose column count is not a multiple of 32 stay unquantized.
    """

    if not _is_matrix_weight(name, tensor) or tensor.shape[1] % Q4_GROUP:
        return {name: tensor}
    values = tensor.float()
    rows = values.shape[0]
    groups = values.reshape(rows, -1, Q4_GROUP)
    minimum = groups.amin(dim=2)
    scale = (groups.amax(dim=2) - minimum).clamp_min(1e-12) / 15.0
    codes = torch.round((groups - minimum.unsqueeze(2)) / scale.unsqueeze(2)).clamp(0, 15)
    codes = codes.to(torch.uint8).reshape(rows, -1, 2)
    packed = (codes[:, :, 0] | (codes[:, :, 1] << 4)).contiguous()
    params = torch.stack((scale, minimum), dim=2).reshape(rows, -1).to(torch.float16)
    return {name: packed, f"{name}.q4_scale": params}


def quantize_state(state: dict[str, torch.Tensor], fmt: str) -> dict[str, torch.Tensor]:
    """Quantize every eligible tensor of a state dictionary to ``fmt``."""

    if fmt == "fp8":
        return {name: quantize_tensor_fp8(name, tensor) for name, tensor in state.items()}
    quantize = quantize_tensor_int8 if fmt == "int8" else quantize_tensor_int4
    if fmt not in ("int8", "int4"):
        raise ValueError(f"format must be one of {FORMATS}")
    output: dict[str, torch.Tensor] = {}
    for name, tensor in state.items():
        output.update(quantize(name, tensor))
    return output


def quantize_checkpoint(source: str | Path, destination: str | Path, *, fmt: str = "fp8") -> Path:
    """Quantize one safetensors checkpoint file."""

    source = Path(source)
    destination = Path(destination)
    if source.suffix != ".safetensors":
        raise ValueError("quantization requires a safetensors checkpoint")
    state = load_file(str(source), device="cpu")
    quantized = quantize_state(state, fmt)
    destination.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {k: v.contiguous() for k, v in quantized.items()},
        str(destination),
        metadata={"format": "pt", "quantization": _format_label(fmt)},
    )
    return destination


def _format_label(fmt: str) -> str:
    return {"fp8": "fp8_e4m3fn", "int8": "int8_rowwise", "int4": "int4_g32"}[fmt]


def quantize_model(
    model_name_or_path: str | Path,
    output_directory: str | Path,
    *,
    fmt: str = "fp8",
    revision: str | None = None,
    tasks: tuple[str, ...] = ("classification", "regression"),
) -> Path:
    """Resolve and quantize every requested task subfolder of an RT-J model."""

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        config, checkpoint = resolve_checkpoint(model_name_or_path, task=task, revision=revision)
        task_directory = output_directory / task
        task_directory.mkdir(parents=True, exist_ok=True)
        checkpoint_name = f"model.{fmt}.safetensors"
        quantize_checkpoint(checkpoint, task_directory / checkpoint_name, fmt=fmt)
        output_config = {
            **config,
            "checkpoint_file": checkpoint_name,
            "quantization": {
                "format": _format_label(fmt),
                "matrix_weights_only": True,
            },
        }
        (task_directory / "config.json").write_text(json.dumps(output_config, indent=2) + "\n")
    return output_directory


def main() -> int:
    parser = argparse.ArgumentParser(description="Quantize an RT-J checkpoint")
    parser.add_argument("model_name_or_path")
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--format", choices=FORMATS, default="fp8", dest="fmt")
    parser.add_argument("--revision")
    parser.add_argument(
        "--task",
        action="append",
        choices=("classification", "regression"),
        dest="tasks",
        help="quantize only this task (repeatable; defaults to both)",
    )
    args = parser.parse_args()
    quantize_model(
        args.model_name_or_path,
        args.output_directory,
        fmt=args.fmt,
        revision=args.revision,
        tasks=tuple(args.tasks) if args.tasks else ("classification", "regression"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
