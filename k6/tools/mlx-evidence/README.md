# mlx-evidence — what these files are, and which of them are byte-exact

Everything here was fetched over HTTP range requests from public repos on
2026-08-29, with no token and no full download. `selftest_mlx_offline.py`
reads all of it; `mlx_surface.py` reads `bf16-shape-census.json.gz` at
runtime (and REFUSES to load a snapshot without it).

| file | what | byte-exact copy of the upstream file? |
|---|---|---|
| `bf16-shape-census.json.gz` | (dtype, shape) for all 38,770 tensors of `zai-org/GLM-5.3-Flash-BF16` @ `a6c167b6`, built from its 62 shard headers | no — derived table |
| `orcarouter-config.json` | `orcarouter/GLM-5.3-Flash-MLX` @ `c80f6810` config.json | **yes** — sha256 `7410d65534e60620abc6e893e6ec089faf818fc3411692b69e668f00029aa958` |
| `orcarouter-index.json.gz` | that repo's `model.safetensors.index.json` | content yes, bytes no — re-serialized compactly (`sort_keys`, no spaces). The upstream file's own sha256 is `cacc0f91d635087c63adf6136e51935e251f12b810fbba2f77551c7aaececa1a` (11,529,499 B) |
| `orcarouter-shard-headers.json.gz` | the safetensors JSON header of each of the 62 shards, keyed by shard name, each with its `__header_len__` | content yes, bytes no — re-serialized |
| `pipenetwork-4bit-config.json` | `pipenetwork/GLM-5.3-Flash-MLX-4bit` @ `f94fd5d1` config.json | **yes** |
| `pipenetwork-4bit-index.json.gz` | that repo's index — the REFUSAL fixture (fused `switch_mlp`, `language_model.model.*`, no MTP layer) | content yes, bytes no. Upstream sha256 `2447ea5bee1874e6b6d79a46320135369d3602dffc68ae43a5528483988b4208` |
| `real-dequant-fixtures/*.npz` | the first 64 output rows of a real quantized tensor (`weight`/`scales`/`biases`) plus the OUTPUT BITS `mlx.core.dequantize` 0.32.2 produced for those rows, with `bits`, `group_size`, `module`, `repo`, `revision` | the tensor rows are byte-exact ranged reads |

An identity computed from a re-serialized index will NOT equal one computed
from the upstream file — which is correct and harmless: `load_mlx_surface`
hashes whatever root it is pointed at, and a real capture is pointed at the
real snapshot. The upstream digests are recorded above so that binding can be
checked by hand.

The fixtures exist so the mlx-equality proof replays where mlx cannot be
installed (every CUDA box). Affine packing is per output row, so a row prefix
is a self-contained, independently decodable tensor.

Regenerate (macOS, mlx installed):

```bash
python mlx_surface.py fetch-meta --repo orcarouter/GLM-5.3-Flash-MLX \
    --revision c80f6810b1a95b5be9042761becc6aa78d189782 --out /tmp/mlx-meta
python mlx_surface.py crosscheck --mlx-root /tmp/mlx-meta \
    --repo orcarouter/GLM-5.3-Flash-MLX --revision c80f6810... \
    --module model.language_model.layers.3.mlp.experts.0.gate_proj \
    ... --save-fixture-slice mlx-evidence/real-dequant-fixtures --fixture-rows 64
python mlx_surface.py crosscheck-raw \
    --repo pipenetwork/GLM-5.3-Flash-MLX-mixed-4_8bit \
    --revision d43ea8b407ce4e9c25e6ac9baec3feab70d9f5f3 \
    --module language_model.model.embed_tokens --rows 64 \
    --save-fixture-slice mlx-evidence/real-dequant-fixtures
```
