from __future__ import annotations

import torch
from relational_transformers.checkpoints import load_state
from safetensors.torch import load_file, save_file

from relational_transformers_utils.quantization import (
    quantize_checkpoint,
    quantize_state,
    quantize_tensor_int4,
    quantize_tensor_int8,
)


def _state() -> dict[str, torch.Tensor]:
    torch.manual_seed(11)
    return {
        "blocks.0.attns.col.wq.weight": torch.randn(64, 64),
        "enc_dict.number.weight": torch.randn(64, 1),
        "enc_dict.number.bias": torch.randn(64),
        "norm_out.scale": torch.ones(64),
    }


def test_int8_round_trips_through_loader(tmp_path):
    path = tmp_path / "model.safetensors"
    save_file({k: v.contiguous() for k, v in quantize_state(_state(), "int8").items()}, str(path))
    loaded = load_state(path)
    original = _state()
    weight = "blocks.0.attns.col.wq.weight"
    error = (loaded[weight] - original[weight]).abs()
    assert error.max() < 0.05
    assert torch.equal(loaded["enc_dict.number.bias"], original["enc_dict.number.bias"])


def test_int4_round_trips_through_loader(tmp_path):
    path = tmp_path / "model.safetensors"
    save_file({k: v.contiguous() for k, v in quantize_state(_state(), "int4").items()}, str(path))
    loaded = load_state(path)
    original = _state()
    weight = "blocks.0.attns.col.wq.weight"
    error = (loaded[weight] - original[weight]).abs()
    assert error.max() < 0.35
    assert error.mean() < 0.12


def test_int4_skips_narrow_matrices():
    result = quantize_tensor_int4("enc_dict.number.weight", torch.randn(64, 1))
    assert set(result) == {"enc_dict.number.weight"}
    assert result["enc_dict.number.weight"].dtype == torch.float32


def test_int8_scales_have_one_entry_per_row():
    result = quantize_tensor_int8("w.weight", torch.randn(8, 16))
    assert result["w.weight"].dtype == torch.int8
    assert result["w.weight.q_scale"].shape == (8,)


def test_fp8_checkpoint_stores_fp8_matrices(tmp_path):
    source = tmp_path / "in.safetensors"
    save_file({k: v.contiguous() for k, v in _state().items()}, str(source))
    destination = quantize_checkpoint(source, tmp_path / "out.safetensors", fmt="fp8")
    raw = load_file(str(destination))
    assert raw["blocks.0.attns.col.wq.weight"].dtype == torch.float8_e4m3fn
    assert raw["enc_dict.number.bias"].dtype == torch.float32
    loaded = load_state(destination)
    assert loaded["blocks.0.attns.col.wq.weight"].dtype == torch.float32
