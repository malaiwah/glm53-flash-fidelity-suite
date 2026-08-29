# Synthetic digests used in the worked examples

Every value below is `sha256("synthetic:" + label)`. They exist so the example
manifests are schema-valid and genuinely self-sealing. Reproduce any of them with:

```
python3 -c "import hashlib;print(hashlib.sha256(b'synthetic:<label>').hexdigest())"
```

| label | value |
|---|---|
| `checksums.txt model card x_fidelity.fidelity_dataset.dataset_sha256 on malaiwah/GLM-5.3-Flash-TR3-6bpw and the registry rows measurement--glm53.k6-6bpw*.brandonmusic-final25` | `f17d9b961ff07f68a4481e598a05b6e7fbc5e879b27e6c6a69aedc8ce99aa7dd` |
| `checksums.txt model card x_fidelity.fidelity_dataset.dataset_sha256 on zai-org/GLM-5.3-Flash-BF16 (proposed) and registry reference--malaiwah.glm53-bf16-root.final25` | `57097646ce891277c765f462f1985a4543d2ae694513eebf87a1e2a3f3c0f40b` |
| `final_norm tensor content` | `796f0954c878021f5f83a9f4ff5543278fd9162da82b640beff475e1aad9a6dc` |
| `k6 backend_identity d19c049f` | `62e02115883eaedcd76bce5a3b1f0ecc37e184b202deb661d4825a2abb3ab6e5` |
| `k6 capture/manifest.json` | `7d87de942700266ca408a02f33dcf93d687e6f26e5fdb3431c08d2139f0a89c8` |
| `k6 capture_content_digest` | `29e5e0ed2004e2d83fa561a590bef42dfceba301b4ff1ea94c9593c394cbdde0` |
| `k6 checkpoint identity a8668be3` | `fb8540afe31f193c211d2d66aafd30b62ecd6e7882308834c3c00f0840a25b43` |
| `k6 config.json` | `a2e2b049f5123aefdce630e744ee65d4ac52f5c10d2c904aedcf44dd61654bd7` |
| `k6 hf revision` | `90dd7d6c86db8015f66566a9fffe4eaa43e3644f4abd6400011fe081e6568526` |
| `k6 model.safetensors.index.json` | `83b72ab2d330e01c845436a6da71080987e840d7874523da2a4e0f3d093c58f6` |
| `k6 runtime_reader 1ccce446` | `c6aa716ffe551cf08b76fd11a1ac5fb269bd4a9e9880772d8cbab6b4650fffac` |
| `k6 upstream capture-receipt 524f017f` | `7d7a9c4c992db9fdccaea2b8c58fb745303089ec4aa82fe2300fda17c458af99` |
| `k6 upstream hidden-capture` | `79e9245a77802b32936a327ab5465f52370f7f78085e254ae064ab87699aea66` |
| `k6 upstream materialization-receipt 3cb08d4d` | `11160ad97aef0709807a1e158746dcf689e19f6f08e8d4a2e406eb5dd876890b` |
| `lane_identity sealed-ep8` | `c8c8f354bfe5fed0cafcf119dab386407e6cdddbc5fef2da6802addef1c7d9bf` |
| `lane_identity streaming` | `124595429ed3406efa6d889ae7242a57fd594264414f5c091f5ca102950e4a13` |
| `native bf16 checkpoint identity` | `2eda2845a93b247685e060c5b366561e9aad94b315523c8920fcac297dfdc90f` |
| `panel/panel.json` | `980c41fef0c5aec4be2beaf43b42f774300de0076ac3a68b399fe36b38fd708d` |
| `root backend_identity` | `0592871a558423243cf8034ca167ff268a2a21561468aa89249ff05566961513` |
| `root capture/manifest.json` | `b80d0d8bacc69cb28bd30a86a10aabca8e993bf8202a8a6c6f2b595c3e4d35bb` |
| `root capture_content_digest` | `5c57d0e94abb1b848e7d02b8202b466333b0bc011e982e935e4458c2eac22132` |
| `root replay-qualification.json` | `c7c4dcb68471dc9a75ccc7bdfcbbad21b0a377b5e4209eb2d47e8affce986b12` |
| `root runtime_reader` | `b53072c325095e4381a85ec5b67397a7fb033fee66e8439f9e40e844b4823809` |
| `root upstream backend` | `381e30926386f3cdca09997f9e0a2e341d9594e9fad355dbab36a73461d8a114` |
| `root upstream capture-receipt` | `57fc8014c344bd59cd6e91807c7a36f5e3156b466fbceebd37923d5d3d97e3ab` |
| `runtime/capture-runtime.json k6 streaming tp4` | `3a42d1153927b33a2ee0c7b01d61ec89436709fffb55aa069dbcdb0b89f91f51` |
| `runtime/capture-runtime.json root bf16 sealed-ep8` | `2bfd0dc30f44e5135ac765ded5a2506e017d435634915421d8e83734a1ec2f4f` |
| `stack_fingerprint k6 streaming tp4` | `0ddc78e6ec7fae403019b2e87c7ded31d3f10271bbb5c7b73a1513798bcdac00` |
| `stack_fingerprint root bf16 sealed-ep8` | `8b1c2b9df2c606b4730d0503a61c965f77ae778c2cb3ea480b6782d9251a3a0e` |
| `suite_token_hash_sha256 final25 k3-preimage` | `66cc7026a9a0cb9dbe192b4ff76c42abfe66a594c35c447779bf1b134b9569ac` |
| `zai bf16 config.json` | `4672cecc91e47bb76d6040f06307347d8caed0844e74eac0b84f698d46bc41e2` |
| `zai bf16 model.safetensors.index.json` | `b72513ac00dbf16a197d4f6d2100143cd2b78499b6832f01d06a55cf0a9e29b1` |

Everything **not** listed here is a real value read from a published artifact,
a registry row, or a file in this repository.
