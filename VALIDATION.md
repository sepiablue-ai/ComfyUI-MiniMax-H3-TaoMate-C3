# 配布用パッケージの検証 / Distribution validation

## 新規cloneからの実生成 / Fresh-clone generation: 2026-09-30

新規取得したComfyUI **0.38.0**（`fb2315f11db0ebfaafa9099a5df5227dc6bb42bc`）へ、このパッケージの `setup.bat` でC3と固定依存ノード2個を導入しました。通常環境と同じWindowsユーザー・専用venv・GPU 0（RTX 4070）を使用し、通常ComfyUIは変更していません。モデル5個は既存ファイルを `extra_model_paths.yaml` で参照し、新規ダウンロードは0個です。50個の未マージFC1 shardを全SHA照合し、ハードリンクで再利用しました。

| Check | Result |
|---|---|
| Git checkout with `core.autocrlf=true` | PASS. `.gitattributes` preserves every packaged file and its fingerprint. |
| Dependency line endings | PASS. LF/CRLF Python sources accepted; source edits and changed DLL bytes rejected. |
| Installer regression tests | **19 passed**, including the two added Git/line-ending checks. |
| `setup.bat -CheckOnly` | PASS. Pinned dependencies, all models by size, all 50 FC1 shard hashes, example input. |
| Node/GUI/API validation | PASS. 22 API nodes, lazy imports, GUI dynamic-combo values, native INT4 GPU/CPU reload and H3 DLL ABI 4. |
| Actual sample workflow | **1 submission, 1 successful video**. `workflows/TaoMate_C3.api.json`, seed 43, 3-step `res_multistep`, unchanged model selections and generation settings. |
| Output | 1024×1792, 124 frames, 24 fps, 5.167 seconds, AAC audio at 32000 Hz. |
| Full FFmpeg decode | PASS for video and audio. |
| Cold prompt duration | **172.29 seconds**, including model loading; client wall time 172.535 seconds. Single installation smoke test, not a warm speed comparison. |
| Cleanup | Task-owned server stopped; validation artifacts remain outside the distribution folder. |

The fresh ComfyUI checkout reused the existing dedicated Python environment without pip changes: Python 3.13.13, Torch 2.14.0+cu130, comfy-kitchen 0.2.35, comfy-aimdo 0.5.5, triton-windows 3.8.0.post28. ComfyUI 0.38.0 lists comfy-kitchen 0.2.36, templates 0.11.70 and embedded docs 0.5.13 in its requirements; this actual successful run used kitchen 0.2.35, templates 0.11.66 and docs 0.5.12. This is evidence for the recorded combination, not a clean installation of every latest Python dependency.

Source workflow SHA-256: `805861acf8314357244d4a96718185e56c2bb5572a5c4e6696dde8c3717eca80`; the environment is recorded in [compatibility.json](compatibility.json). Output SHA-256: `db2fef05f0237e69a2ad7894345d584186999ea6b79134d2f1ecef56e87ddced`. Prompt ID: `19148c71-45f4-41c2-8cfc-a8d5571e8ccd`.

Not tested in this run: fresh 50-layer FC1 conversion, newly provisioned Python environment, model downloads, browser drag-and-drop, other GPUs, or subjective quality acceptance. Normal setup now automatically starts the first FC1 conversion when all five models are present and the cache is absent; the tested path reused an existing cache. GUI/API validation and actual API generation use the same supplied workflow settings.

## 初回配布検証 / Initial packaging checks: 2026-09-28

以下の表と未実施項目は2026-09-28時点の記録です。実生成の追加結果は上記を参照してください。

## 確認したこと / Verified

| Check | Result |
|---|---|
| Empty custom_nodes → setup.bat installation | PASS. H3-Optimizations and FastVAE fetched from the exact GitHub commits in dependency-lock.json. No global pip or core updates. |
| Windows Japanese + space-containing destination | PASS. Tested in a project-local disposable ComfyUI directory. |
| Dependency integrity | PASS. 110 H3 files, including Windows DLL and BUILD_ID, plus 3 FastVAE runtime files match the pinned fingerprints. |
| Existing FC1 cache reuse | PASS. All 50 FC1 SHA-256 values verified; new FC1-only manifest; Gate omitted. Source cache manifest preserved. |
| Real ComfyUI node registration | PASS. All 22 graph nodes / 21 distinct types available. No KJNodes or old W4A4-VSA pack in the isolated custom_nodes directory. |
| Dependency-first / C3-package-first load order | PASS in separate processes. Lazy implementation import and canonical h3_optimizations identity verified. |
| Actual ComfyUI validate_prompt | PASS for the API graph and GUI SaveVideo dynamic-combo settings. No execution submitted. |
| GUI/API contract | PASS. Links, widget values, fixed seed, prompt, model settings and numerical parameters checked. SaveVideo uses named dynamic-combo widgets compatible with the inspected ComfyUI frontend. |
| GPU native INT4 check | PASS on RTX 4070. Native cutlass_int4_dequant called, finite output, GPU→CPU→GPU output exact. |
| H3 DLL load / ABI | PASS. ABI 4, pinned native-v11 DLL SHA-256. |
| Adopted C2/C3 source bytes | PASS. runtime/h3_c2_held_ffn8.py and runtime/h3_c3_held_out.py unchanged; cache.py and convert.py also unchanged. |
| Installer/packaging regression tests | **17 passed**. Global-Python rejection; duplicate nodes; modified upstream code and DLL override; preservation of existing files; failed-fetch cleanup; cache tamper/path traversal; FC1-only reuse; premerged-LoRA/Ref2VA rejection; read-only check with missing input directory; cold registration; legacy migration; GUI/API consistency. |

ComfyUI 0.37.0の追跡済みソースを一時フォルダへコピーし、既存の専用venvを**読み取り利用**して検証しました。標準ComfyUIのcustom_nodes・起動bat・共有モデル・Pythonパッケージは変更していません。GPUスモークテストは明示的にGPU 0を使用しました。サーバーは起動していません。

The installation target was isolated. The existing dedicated ComfyUI venv was reused without installing or upgrading Python packages. Standard custom nodes, launchers and model files were not modified. No server was started.

## 未実施の範囲 / Limits

- **この配布フォルダからの動画生成は0本。** 採用時の動画・音声一致と速度結果は既存実験に基づきます。今回、配布版の新しい速度値や再生成による品質合格を主張しません。
- 新規50層変換は今回再実行せず、既存キャッシュを全SHA照合して再利用しました。新規変換コードは採用時の原本をそのまま収録しています。
- venvを完全新規作成して全Python依存をダウンロードする検証はしていません。モデルファイルも新規ダウンロードしていません。
- GUIはJSON構造・値・リンク・実バックエンド検証まで確認しました。ブラウザでのドラッグ＆ドロップ操作は今回実施していません。
- Windows PortableのPython自動検出はコードに含みますが、今回の実インストールはvenv版です。Linux、他GPU、異なるTorch/Kitchen/H3-Optimizations版での実生成は未検証です。

No new full-video generation, fresh 50-layer conversion, clean Python-environment provisioning, browser drag-and-drop test, or model download was performed for packaging. The adopted arithmetic sources are unchanged, but the above installation checks are not a substitute for a new end-to-end quality/performance comparison on another machine.

## 再実行 / Repeat the checks

Run from the package directory using the target ComfyUI Python:

```bat
"C:\AI\.venv\Scripts\python.exe" -B -X utf8 -m unittest discover -s tests -v
setup.bat -ComfyRoot "C:\AI\ComfyUI" -Python "C:\AI\.venv\Scripts\python.exe" -CheckOnly
"C:\AI\.venv\Scripts\python.exe" -B -X utf8 tools\check_runtime.py --comfy-root "C:\AI\ComfyUI" --package-first
"C:\AI\.venv\Scripts\python.exe" -B -X utf8 tools\check_runtime.py --comfy-root "C:\AI\ComfyUI" --native --gpu 0
```

`-CheckOnly` performs no writes. Unit tests use temporary folders (default `tmp` inside this package; override with process-local `C3_TEST_TMP`) and remove their test contents. A native check performs a small GPU operation, not a video run.

Maintainers editing distribution files should review the changes, rerun relevant checks and update `package-files.json` with `tools/seal_package.py`. This inventory checks copy integrity; it is not a cryptographic signature or permission to bypass dependency validation.
