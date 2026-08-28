"""Exercise dione_surface.audit_slice_placement itself on the REAL fetched
payloads: builds a mini dione shard + mini bf16 root from the ranged bytes,
then calls the very function the morning capture gates on."""
import json, sys
from pathlib import Path
import numpy as np, torch
from safetensors.torch import save_file

TOOLS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOLS))
import dione_surface as ds

HERE = Path(__file__).resolve().parent
PRE = 'model.language_model.layers.3.mlp.experts.0.'

def load(name, np_dtype, shape, view=None):
    raw = (HERE / 'payloads' / f'{name}.bin').read_bytes()
    a = torch.from_numpy(np.frombuffer(raw, dtype=np_dtype).reshape(shape).copy())
    return a.view(view) if view else a

mini = HERE / 'mini-real'
(mini / 'layers').mkdir(parents=True, exist_ok=True)
(mini / 'bf16').mkdir(exist_ok=True)

tensors = {}
for proj in ('gate_proj', 'up_proj', 'down_proj'):
    geo = ds.expected_slice_geometry(proj, bits=4, tp_size=4)
    for r in range(4):
        tensors[ds.slice_name(3, 0, proj, r, 'trellis')] = load(PRE + f'{proj}.rank{r}.trellis', np.int16, geo['trellis'][1])
        tensors[ds.slice_name(3, 0, proj, r, 'suh')] = load(PRE + f'{proj}.rank{r}.suh', np.uint16, geo['suh'][1], torch.float16)
        tensors[ds.slice_name(3, 0, proj, r, 'svh')] = load(PRE + f'{proj}.rank{r}.svh', np.uint16, geo['svh'][1], torch.float16)
        tensors[ds.slice_name(3, 0, proj, r, 'mcg')] = load(PRE + f'{proj}.rank{r}.mcg', np.int32, ())
shard_rel = 'layers/mini.safetensors'
save_file(tensors, str(mini / shard_rel))

official = {}
for proj, shape in (('gate_proj', (2048, 4096)), ('up_proj', (2048, 4096)), ('down_proj', (4096, 2048))):
    official[ds.official_name(3, 0, proj)] = load(PRE + f'{proj}.weight', np.uint16, shape, torch.bfloat16)
save_file(official, str(mini / 'bf16' / 'mini.safetensors'))
(mini / 'bf16' / 'model.safetensors.index.json').write_text(json.dumps(
    {'weight_map': {name: 'mini.safetensors' for name in official}}))

surface = ds.DioneSurface(
    root=mini, repo='0xSero/GLM-5.3-Flash-EXL3-Q4',
    revision='99cccdf0e8741715662c383828a9ea601990c125', bits=4, tp_size=4,
    fmt=ds.DIONE_FORMAT, source_repo='zai-org/GLM-5.3-Flash-BF16',
    source_revision='a6c167b62691b2bac901344b65cb651a70f53e43',
    config_sha256='0'*64, index_sha256='0'*64, exl3_manifest_sha256=None,
    weight_map={name: shard_rel for name in tensors}, retained_names=(),
    shard_hash_verification='skipped', text_vocab_size=154880)
shards = ds.DioneShardReader(surface)
audit = ds.audit_slice_placement(surface, shards, mini / 'bf16', layer=3, expert=0)
print(json.dumps(audit, indent=1))
assert audit['passed']
for proj, row in audit['projections'].items():
    print(proj, 'diag', [round(row['cosine_rank_x_block'][i][i], 4) for i in range(4)],
          'rel_l2', round(row['assembled_rel_l2_vs_official_bf16'], 4))
(HERE / 'real-audit-via-adapter.json').write_text(json.dumps(audit, indent=1))
print('REAL AUDIT VIA ADAPTER: PASS')
