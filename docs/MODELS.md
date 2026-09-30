# モデル / Models

モデルは同梱せず、自動ダウンロードもしません。次の固定revisionのファイルを使用します。Models are not bundled or automatically downloaded. Use the pinned files below.

`ComfyUI/models/<category>/` に保存するか、既存の `extra_model_paths.yaml` から参照してください。Store each file under its category or configure extra model paths.

## minimax_h3_fl2va_pruned_int8_convrot.safetensors

- Folder: `diffusion_models`
- Download: [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/bf92c4091e333e69b8ca1998e0a669f15cb0832b/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors)
- Revision: `bf92c4091e333e69b8ca1998e0a669f15cb0832b`
- Bytes: 20970379616 (19.53 GiB)
- SHA-256: `e889202c41dafb67b10d67b97f0d8541508036a6090af23425a5c2615d03c47a`

## qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors

- Folder: `text_encoders`
- Download: [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/bf92c4091e333e69b8ca1998e0a669f15cb0832b/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors)
- Revision: `bf92c4091e333e69b8ca1998e0a669f15cb0832b`
- Bytes: 15687142551 (14.61 GiB)
- SHA-256: `35a88d51044231fe332301d7a62aa81e3f2cba62febeb446e2c1e3e0ef76f2c6`

## minimax_h3_audio_vae_fp32.safetensors

- Folder: `vae`
- Download: [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/bf92c4091e333e69b8ca1998e0a669f15cb0832b/vae/minimax_h3_audio_vae_fp32.safetensors)
- Revision: `bf92c4091e333e69b8ca1998e0a669f15cb0832b`
- Bytes: 605254808 (0.56 GiB)
- SHA-256: `8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48`

## minimax_h3_video_vae_int8_convrot.safetensors

- Folder: `vae`
- Download: [Kijai/MiniMax-H3-experimental](https://huggingface.co/Kijai/MiniMax-H3-experimental/resolve/e042fe480f58806578713532b8ae4e3d47d1bd63/minimax_h3_video_vae_int8_convrot.safetensors)
- Revision: `e042fe480f58806578713532b8ae4e3d47d1bd63`
- Bytes: 3171670912 (2.95 GiB)
- SHA-256: `9bb2d96f218c76babd85e0611b85ca8fb330a90546c01a0005e8a58a59593410`

## TaoMate-H3-3step-ComfyUI.safetensors

- Folder: `loras`
- Download: [CZMartin22/TaoMate-H3-3step-ComfyUI](https://huggingface.co/CZMartin22/TaoMate-H3-3step-ComfyUI/resolve/3fc6fda82a6684543235b38bcd45f075530ed1c6/TaoMate-H3-3step-ComfyUI.safetensors)
- Revision: `3fc6fda82a6684543235b38bcd45f075530ed1c6`
- Bytes: 1240539224 (1.16 GiB)
- SHA-256: `908d6c9a8d4cb7311b9b48aece606a1d29033f07e2ef23676f8404acc07d5173`

## Video VAE の同名ファイル / Same-name VAE files

検証対象はKijai版の3,171,670,912 bytes、SHA-256 `9bb2d96f...` です。Comfy-Orgの同名ファイル（2,811,065,184 bytes、SHA-256 `52a2c8c7...`）へ置き換えると比較条件が変わります。セットアップはこの違いをサイズで検出し、`-VerifyModels`では全SHAを検査します。

The validated Kijai VAE is not interchangeable with the different Comfy-Org file of the same name for reproducing these results. Setup detects the size difference; `-VerifyModels` checks the full hash.

既存のTaoMate事前マージ用重みや残差LoRAは使用しません。元のFL2VAと元のTaoMate LoRA strength 1.0をworkflow内で組み合わせます。No premerged diffusion weights or residual LoRA are used.

モデル5個が揃っていてFC1キャッシュがない場合は、通常の `setup.bat` で初回GPU変換を自動実行します。`-CacheSource` で既存キャッシュを指定すれば変換を省略できます。キャッシュを再利用する場合は元モデルのsafetensorsヘッダとサイズ、および全FC1 shardのSHAを確認します。Full source-model hashes require `-VerifyModels`; the default readiness check uses size and the cache’s source header identity.
