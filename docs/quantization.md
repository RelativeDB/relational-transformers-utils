# Quantization

The quantizers here produce checkpoints that `relational_transformers.checkpoints.load_state`
accepts, so every output loads through the standard constructor. Only two-dimensional
matrix weights are quantized; biases, normalization scales, and mask embeddings keep
their original floating-point type.

## Formats

| Format | Storage | On-disk keys | Portable loading |
| --- | --- | --- | --- |
| `fp8` | `float8_e4m3fn` weights | `<name>.weight` | widened to float32 off CUDA |
| `int8` | int8 weight + one scale per output row | `<name>.weight`, `<name>.weight.q_scale` | dequantized while loading |
| `int4` | two values per byte, groups of 32 | `<name>.weight`, `<name>.weight.q4_scale` | unpacked while loading |

Int4 stores an fp16 `(scale, minimum)` pair per group of 32 values, and a matrix whose
column count is not a multiple of 32 stays unquantized. FP8 can remain packed in the
Triton CUDA backend.

## Command Line

```bash
rt-quantize RelativeDB/rt-j-fp16 ./rt-j-int8 --format int8
```

The command resolves the `classification/` and `regression/` subfolders by default,
writes `model.<format>.safetensors` plus a `config.json` whose `quantization` block
records the format, and accepts `--task` to quantize one subfolder and `--revision` to
pin a Hub revision.

## Python API

```python
from relational_transformers_utils import quantize_model, quantize_state

quantize_model("RelativeDB/rt-j-fp16", "./rt-j-fp8-local", fmt="fp8")
quantized = quantize_state(state_dict, "int4")   # dict in, dict out
```

`quantize_checkpoint` handles a single safetensors file. The test suite round-trips
every format through the core loader and bounds the reconstruction error.

## Validation

Compare logits, ranking, and task metrics against the FP16 checkpoint on representative
contexts before deploying a quantized artifact. `better()` from the
[metrics](metrics.md) page gives the direction-aware comparison, and the core package's
`RUN_HUB_TESTS=1` suite validates the published formats end to end.
