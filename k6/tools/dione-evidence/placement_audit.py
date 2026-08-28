import numpy as np, torch, json
from quant_pipeline.evaluation.glm53_packed_k4_reader import decode_choice_hf
from quant_pipeline.checkpoint.packed_payload import MCG_MARKER_SIGNED_INT32
import torch.nn.functional as F

def load(name, dtype, shape):
    raw = open(f'payloads/{name}.bin','rb').read()
    a = np.frombuffer(raw, dtype=dtype).reshape(shape).copy()
    return torch.from_numpy(a)

pre='model.language_model.layers.3.mlp.experts.0.'
official_gate = load(pre+'gate_proj.weight', np.uint16, (2048,4096)).view(torch.bfloat16).float()
official_down = load(pre+'down_proj.weight', np.uint16, (4096,2048)).view(torch.bfloat16).float()

results={}
dec_gate=[]
for r in range(4):
    tr = load(pre+f'gate_proj.rank{r}.trellis', np.int16, (256,32,64))
    suh = load(pre+f'gate_proj.rank{r}.suh', np.uint16, (4096,)).view(torch.float16)
    svh = load(pre+f'gate_proj.rank{r}.svh', np.uint16, (512,)).view(torch.float16)
    mcg = load(pre+f'gate_proj.rank{r}.mcg', np.int32, ())
    assert int(mcg)==MCG_MARKER_SIGNED_INT32, ('mcg marker', int(mcg))
    d = decode_choice_hf(tr, suh, svh, bits=4)
    assert tuple(d.shape)==(512,4096)
    dec_gate.append(d)
corr=np.zeros((4,4))
for r in range(4):
    for b in range(4):
        corr[r,b]=F.cosine_similarity(dec_gate[r].flatten(), official_gate[b*512:(b+1)*512].flatten(), dim=0).item()
print('gate corr (rank x block):'); print(np.array2string(corr, precision=4))
full=torch.cat(dec_gate,0)
rel=((full-official_gate).norm()/official_gate.norm()).item()
print('gate concat-dim0 rel L2 vs official:', rel)
results['gate_proj']={'corr_rank_x_block':corr.tolist(),'concat_dim':0,'rel_l2_vs_official':rel}

dec_down=[]
for r in range(4):
    tr = load(pre+f'down_proj.rank{r}.trellis', np.int16, (32,256,64))
    suh = load(pre+f'down_proj.rank{r}.suh', np.uint16, (512,)).view(torch.float16)
    svh = load(pre+f'down_proj.rank{r}.svh', np.uint16, (4096,)).view(torch.float16)
    mcg = load(pre+f'down_proj.rank{r}.mcg', np.int32, ())
    assert int(mcg)==MCG_MARKER_SIGNED_INT32
    d = decode_choice_hf(tr, suh, svh, bits=4)
    assert tuple(d.shape)==(4096,512)
    dec_down.append(d)
corr=np.zeros((4,4))
for r in range(4):
    for b in range(4):
        corr[r,b]=F.cosine_similarity(dec_down[r].flatten(), official_down[:, b*512:(b+1)*512].flatten(), dim=0).item()
print('down corr (rank x block):'); print(np.array2string(corr, precision=4))
full=torch.cat(dec_down,1)
rel=((full-official_down).norm()/official_down.norm()).item()
print('down concat-dim1 rel L2 vs official:', rel)
results['down_proj']={'corr_rank_x_block':corr.tolist(),'concat_dim':1,'rel_l2_vs_official':rel}
results['module']='model.language_model.layers.3.mlp.experts.0'
results['dione_revision']='99cccdf0e8741715662c383828a9ea601990c125'
results['bf16_revision']='a6c167b62691b2bac901344b65cb651a70f53e43'
results['decoder']='quant_pipeline.evaluation.glm53_packed_k4_reader.decode_choice_hf (bits=4), CPU fp32'
json.dump(results, open('real-payload-placement-audit.json','w'), indent=1)
print('WROTE real-payload-placement-audit.json')
