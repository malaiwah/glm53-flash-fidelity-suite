---
license: mit
base_model: zai-org/GLM-5.3-Flash-BF16
base_model_relation: quantized
tags:
  - exl3
  - text-generation
  - mixture-of-experts
---

# GLM-5.3-Flash-EXL3-3.0bpw

Campaign status: **pending**

This repository is reserved for a selective 3.0 bpw EXL3 quantization of [zai-org/GLM-5.3-Flash-BF16](https://huggingface.co/zai-org/GLM-5.3-Flash-BF16), pinned at `a6c167b62691b2bac901344b65cb651a70f53e43`.

No model weights are present yet. Conversion, assembly, exact manifest verification, runtime testing, generation testing, and quality evaluation are pending. This card will be replaced only after the complete snapshot passes its local and remote release gates.

Planned scope: routed-expert gate/up/down projections in layers 3-44 will use EXL3 K3. Attention, indexers, mHC, routers, shared experts, dense layers 0-2, embeddings, LM head, norms, vision, and MTP remain in source BF16. The planned artifact retains the custom TP4 selective-EXL3 layout and will require a compatible loader; stock Transformers compatibility is not claimed.

Calibration is shared with the accepted 4.0 bpw control: 1,228,800 tokens with natural top-8 routing, all 42 routed layers and all 288 experts covered, and a minimum route count of 1,655 against a 1,024 floor. The public coverage report will be added with the weights.

Suite: [0xSero/GLM-5.3-Flash-EXL3](https://huggingface.co/0xSero/GLM-5.3-Flash-EXL3)

Credits: Z.AI for the MIT-licensed base model, ExLlamaV3/TurboDerp for EXL3, and Dione for the selective conversion workflow.
