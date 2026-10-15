# ComfyUI MiniMax H3 TaoMate C3

日本語 | [English](README.en.md)

**TaoMate-H3（3stepで生成できるよう蒸留したMiniMax H3）を、RTX 4070 12GBで1024×1792・音声付き約2分27秒で生成するComfyUI最適化パッケージです。** Windowsセットアップとサンプルworkflowを同梱しています。

| FL2VA（先頭画像1枚）・1024×1792・124フレーム | 生成時間 |
|---|---:|
| 4step・W4A4 + Streaming VSA 5%（TaoMateなし） | 218.2秒（3分38.2秒） |
| TaoMate 3step + sparse attention 10% | 183.3秒（3分3.3秒） |
| **本repo：TaoMate 3step + W4A4 + sparse 10% + 重み保持（C2/C3）** | **146.9秒（2分26.9秒）** |

[さやかベンチマーク](https://sayakabenchmark.pages.dev/)のFL2VAタブ、case01の実測です（2026-09-30、各1回）。上から順にFL2VA [018](https://sayakabenchmark.pages.dev/benchmarks/fl2va/018-w4a4-streaming-vsa5-i2v/workflow.json)、[020](https://sayakabenchmark.pages.dev/benchmarks/fl2va/020-taomate-h3-fl2va-i2v-sparse10/workflow.json)、[021](https://sayakabenchmark.pages.dev/benchmarks/fl2va/021-taomate-h3-fl2va-i2v-c3-fastest/workflow.json)のworkflowです（番号はサイト上の登録番号）。2行目から3行目で約20%短縮していますが、動画VAEの復号方式の違い（通常VAE→FastVAE）も含みます。

- **参考：** 544×960で生成し、[ComfyUI-H3-X2-Stream](https://github.com/sepiablue-ai/ComfyUI-H3-X2-Stream)で1088×1920として保存する構成（FL2VA [016](https://sayakabenchmark.pages.dev/benchmarks/fl2va/016-taomate-3step-c3-sparse10-x2vae-int8-stream/workflow.json)）は51.7秒でした。このworkflowは本repoには同梱していません。

**用語**

| 用語 | 意味 |
|---|---|
| TaoMate 3step | Alibaba TaoLiveのLoRAで、H3を3stepで生成できるようにしたもの |
| FC1 W4A4 | 各層のFFN前半（FC1）の重みと計算を4bitにする。変換は導入時に一度だけ |
| sparse attention 10% | 重要なattentionだけを計算する（約10%を保持） |
| C2（Held FFN8） | FFNを8分割で処理し、その重みを必要な範囲で保持して転送やcastを減らす |
| C3（Held Attention Out） | attention出力層のINT8重みも同様に保持する。C2からさらに約3%短縮 |

> 作者：かみもと（Kamimoto）｜検証結果の速報は **[X（@sep_is_heim）](https://x.com/sep_is_heim)**、解説記事は **[note](https://note.com/sepiablue)**（[記事INDEX](https://note.com/sepiablue/n/n8e612c619a3e)）で公開しています。

Windows用の `setup.bat`、固定版の依存ノードを取得するインストーラー、GUI/API workflow、同じ検証用先頭画像を収録しています。モデルと変換キャッシュは別途必要です。配布フォルダ単独で導入でき、元の実験プロジェクトは不要です。

通常のH3 sparse attentionを使用します。旧リポジトリのStreaming VSA 5%とは別の構成です。画像から動画を作る **FL2VAモデルのI2V** を検証しており、Ref2VAのベンチマークとは区別してください。

## 成果の帰属とこのパッケージの役割

このパッケージは、以下の作者・プロジェクトの成果を利用しています。

| 作者・プロジェクト | 利用している成果 |
|---|---|
| [MiniMax AI](https://huggingface.co/MiniMaxAI/MiniMax-H3) | MiniMax H3の基盤モデルと重み |
| [Alibaba TaoLive AIGC Team](https://huggingface.co/TaoLiveAIGC/TaoMate-H3) | TaoMate-H3の3-step蒸留LoRA |
| [CZMartin22](https://huggingface.co/CZMartin22/TaoMate-H3-3step-ComfyUI) | TaoMate LoRAのComfyUI形式への変換 |
| [Zironic / H3-Optimizations](https://github.com/Zironic/H3-Optimizations) | sparse attention、メモリ最適化、HeldMLP / INT8処理の基盤 |
| [MotionCache-FastVAE](https://github.com/Mozer/ComfyUI-MiniMax-H3-MotionCache-FastVAE) | 動画VAEの高速復号 |

本repoでは、既存の基盤を利用した**FC1 W4A4キャッシュの導入・ロード、FFN8での重み保持（C2）、attention-outの独立INT8コピー保持（C3）**を、導入・実行できる構成にまとめています。固定依存版のセットアップ、キャッシュ再利用、GUI/API workflow、入力画像、RTX 4070での実測条件を収録しています。TaoMateの学習・蒸留、ComfyUI形式へのLoRA変換はそれぞれ上記作者の成果です。[出典とライセンス](THIRD_PARTY_NOTICES.md)も参照してください。

## RTX 4070 12GBでの実測

同梱サンプルは1024×1792、124フレーム、24fps、seed 43、3-step `res_multistep`です。

| 検証 | 実測結果 |
|---|---:|
| 同一条件のC2、通常cache warm | 149.392秒 |
| 同一条件のC3、通常cache warm | **145.021秒** |
| C3追加による短縮 | **4.371秒 / 約2.93%** |
| 2026-09-30、新規ComfyUI 0.38.0での初回生成（モデル読込込み） | **172.29秒** |

採用時は計10本の比較動画を使用し、同seedのC2/C3間で全復号映像とPCMが一致しました。短縮率と一致確認は**C2→C3の差分**に対する結果です。最適化なしのTaoMateに対する全体の短縮率や、dense H3との同一画質を示すものではありません。warm値と初回値は条件が異なります。

新規ComfyUIへの `setup.bat` 導入では、同梱API workflowを1回送信して動画1本の成功、AAC音声、映像・音声のFFmpeg全復号を確認しました。既存の専用venvとFC1キャッシュを再利用し、新規モデルダウンロードは行っていません。完全新規のPython環境構築と50層キャッシュ新規変換はこの確認の対象外です。[比較結果](docs/RESULTS.md)、[導入・実生成記録](VALIDATION.md)、[環境と版](compatibility.json)に詳細を記載しています。

## 導入

1. MiniMax H3が動作するComfyUIと、その専用venvまたはPortable版のPythonを準備し、[モデル一覧](docs/MODELS.md)の5ファイルを配置します。検証環境はWindows 11、RTX 4070 12GBです。[互換性情報](compatibility.json)に正確な版を記録しています。
2. このrepoをcloneするかZIPを展開します。`ComfyUI/custom_nodes`の中でも外でも構いません。Git for Windowsは不足する依存ノードの取得に使用します。
3. ComfyUIを終了し、`setup.bat`を実行します。ダブルクリックではComfyUIの場所を尋ねます。指定する場所は**`main.py`があるディレクトリ**です。Python候補が複数ある場合は下記のように明示します。

```bat
setup.bat -ComfyRoot "C:\AI\ComfyUI" -Python "C:\AI\.venv\Scripts\python.exe"
```

PowerShellからは `setup.bat` を ` .\setup.bat` として実行してください。Portable版の例は `-ComfyRoot "C:\AI\ComfyUI_windows_portable\ComfyUI" -Python "C:\AI\ComfyUI_windows_portable\python_embeded\python.exe"` です。

セットアップはC3ノードと以下の依存ノードを導入し、検証画像を `ComfyUI/input/taomate_c3/first_frame.png` にコピーします。モデル5個が揃っていてFC1キャッシュがない場合は、初回だけGPUでキャッシュを自動作成します。他の重いGPU処理を止めてから実行してください。モデルが不足していてもノードの導入は完了し、最後に **GENERATION NOT READY** と不足物を表示します。生成できる状態かどうかは後述の `-CheckOnly` で判定してください。

| 依存ノード | 固定コミット | 用途 |
|---|---|---|
| [H3-Optimizations](https://github.com/Zironic/H3-Optimizations) | `862774944a331bc1a66cee1cf805b94bd138ebbe` | H3MemoryOptimization / H3SparseAttention、ネイティブINT8処理 |
| [MotionCache-FastVAE](https://github.com/Mozer/ComfyUI-MiniMax-H3-MotionCache-FastVAE) | `b719329e0ecf35f0ae08d241c363ed1e56adbb95` | MiniMaxH3FastVAEDecodeのみ使用 |

このworkflowではKJNodes、旧W4A4-VSAパッケージ、VSA Gateモデルは不要です。依存ノードはGitHubから指定コミットを取得し、PythonコードとWindows DLLを[指紋一覧](dependency-lock.json)で照合します。既存の異なる版を自動更新・リセットしません。

Gitでcloneした場合も `.gitattributes` で配布ファイルの元のバイト列を保持します。依存ノードのPythonコードはLF/CRLFの違いだけを許容し、DLLはバイト単位で照合します。

## モデルとFC1キャッシュ

[モデル一覧と固定ダウンロード先](docs/MODELS.md)にある5ファイルを用意してください。既存の `extra_model_paths.yaml` も読み取ります。

- Diffusion: `minimax_h3_fl2va_pruned_int8_convrot.safetensors`
- LoRA: `TaoMate-H3-3step-ComfyUI.safetensors`
- Text encoder: `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`
- Video VAE: `minimax_h3_video_vae_int8_convrot.safetensors`（**検証したKijai版**）
- Audio VAE: `minimax_h3_audio_vae_fp32.safetensors`

モデルは自動ダウンロードしません。同名のVideo VAEでも配布元によって内容が違います。セットアップでサイズ、`-VerifyModels`で全SHA-256を検査できます。

FC1は初回だけオフライン変換します。生成中に重みをW4A4へ変換する設計ではありません。出力先は `ComfyUI/models/h3_preconverted/taomate_c3_fc1`、約 **3.86 GB / 3.59 GiB** です。

**既存のC3用キャッシュを再利用する場合：**

```bat
setup.bat -ComfyRoot "C:\AI\ComfyUI" -Python "C:\AI\.venv\Scripts\python.exe" -CacheSource "D:\H3-cache\fl2va_fc1_gate"
```

50個のFC1 shardを全SHA-256照合し、同一ボリュームではハードリンク、別ボリュームではコピーします。新しいmanifestはFC1専用で作成し、元manifestを変更しません。Gateはコピーしません。共有されたshardを後から上書き編集しないでください。LoRA事前マージ版やRef2VA用のキャッシュは使用できません。

**新しく変換する場合（モデル配置後の通常setupでも自動実行）：**

```bat
setup.bat -ComfyRoot "C:\AI\ComfyUI" -Python "C:\AI\.venv\Scripts\python.exe" -BuildCache -Gpu 0
```

この操作はGPUを使用します。動画生成や他の重いGPU処理を止めてから実行してください。50層を変換し、native INT4・保存/再読込・CPU offload/reloadの一致を検査します。途中で失敗したキャッシュを完成品として配置しません。`-Gpu`は子プロセスだけに適用されます。

## 動作確認と実行

```bat
setup.bat -ComfyRoot "C:\AI\ComfyUI" -Python "C:\AI\.venv\Scripts\python.exe" -CheckOnly
"C:\AI\.venv\Scripts\python.exe" -B -X utf8 tools\check_runtime.py --comfy-root "C:\AI\ComfyUI" --native --gpu 0
```

`-CheckOnly` はファイルを変更せず、固定依存、モデルサイズ、FC1全shardのSHA-256、検証画像を確認します。`-VerifyModels`を併用すると大きな元モデル5個も全SHA-256照合するため、時間がかかります。`check_runtime.py`は実際のComfyUIノード登録・22ノードのAPI検証・遅延importを確認します。`--native`は小さなGPU INT4往復検査とH3 DLLのABI検査も行います。サーバーや動画生成は開始しません。

ComfyUIを普段の起動方法で起動し、[TaoMate_C3.json](workflows/TaoMate_C3.json)を読み込んでください。API/ベンチマーク用は [TaoMate_C3.api.json](workflows/TaoMate_C3.api.json) です。モデルをサブフォルダに保存している場合は、ノードのモデル選択を実際の相対名に合わせます。検証プロンプトと画像を同梱しています。別の画像・プロンプトにも変更できますが、検証時の画質や時間の保証はありません。

| 設定 | 同梱workflow |
|---|---|
| サイズ / フレーム / fps | 1024×1792 / 124 / 24 |
| Seed / sampler | 43 / res_multistep |
| Sigmas | `1.0, 0.9611650705337524, 0.8533332943916321, 0.0` |
| Sigma shift | video 12 / audio 3 |
| TaoMate LoRA | strength 1.0、未マージの元モデルに適用 |
| Sparse | budget 0.1、denser early/late OFF、4096 rows |
| MLP / attention out | Held FFN8、threshold 4096 / C3 INT8保持 |
| FastVAE | tile_batch_size 2 |

通常cacheでwarm時間を測るときは、1本目seed 43→2本目seed 44にしてsamplerを再実行させます。同じseedで全ノードがcache命中した時間を生成速度として扱わないでください。採用時は `--disable-pinned-memory` を使用しています。`--cache-none`や`--fast fp16_accumulation`を既定起動に追加しません。

## 既存C3 / ベンチマークからの移行

このパッケージが登録するIDは `H3TaoMateW4A4Loader`、`H3C2HeldFFN8`、`H3C3HeldOut` の3個です。古いC3 API workflowの `H3V2PreconvertedLoader` は旧W4A4-VSAが提供するIDのため、そのままでは新パッケージ単独で使えません。同梱API workflowへ置き換えるか、コピーを変換します。

```bat
"C:\AI\.venv\Scripts\python.exe" -B -X utf8 tools\migrate_workflow.py "old_c3.api.json" "portable_c3.api.json"
```

このツールはローダーID、キャッシュパス、旧検証画像名だけを置き換えます。seed・prompt・数値設定は保持し、入力ファイルを上書きしません。VSAやC2だけのworkflowをC3に変換するツールではありません。

手動で導入した `ComfyUI-Sayaka-H3-C2C3` や単独の `h3_c2_held_ffn8.py` / `h3_c3_held_out.py` と併用するとIDが重複します。セットアップは検出して停止します。ComfyUIを終了後、バックアップして `*.disabled` にリネームするか、active custom_nodesの外に移してから実行してください。セットアップが勝手に削除することはありません。`extra_model_paths.yaml`で参照するcustom_nodes内の重複も対象です。

## 検証範囲

採用時の同一条件比較では、C2→C3の通常cache warmが **149.392→145.021秒（約2.93%短縮）**。10本の比較動画で、同seedの復号映像とPCMが一致しました。これはC2に対するC3の保持処理の評価であり、TaoMate・W4A4・sparse化全体を元の密なH3と同一画質とする主張ではありません。後日の別セッションではC3 warm **143.521秒** を記録しました。条件を跨いだ保証値ではありません。

今回の配布確認と未実施項目は [VALIDATION.md](VALIDATION.md)、採用結果と除外した案は [RESULTS.md](docs/RESULTS.md) を参照してください。採用コードの原本SHA-256は [source-provenance.json](docs/source-provenance.json) に保存しています。

セットアップはグローバルPythonへのpip、ComfyUI本体・Torch・Kitchenの自動更新、標準起動bat・共有モデル・ACLの変更を行いません。依存の固定版を更新するときはC3の内部APIとの整合を再検証してください。

## 作者・関連リポジトリ

かみもと（Kamimoto）です。RTX 4070 12GBで、MiniMax H3とComfyUIの高速化・省VRAM化を実測しながら公開しています。

- X：[@sep_is_heim](https://x.com/sep_is_heim) — step数・秒数・短縮率などの検証速報
- note：[かみもと｜ローカルAI生成](https://note.com/sepiablue) — 導入方法と実測の解説記事（[記事INDEX](https://note.com/sepiablue/n/n8e612c619a3e)）
- [さやかベンチマーク](https://sayakabenchmark.pages.dev/) — H3ワークフローの速度・画質比較

| リポジトリ | 内容 |
|---|---|
| [ComfyUI-H3-X2-Stream](https://github.com/sepiablue-ai/ComfyUI-H3-X2-Stream) | X2 VAEのINT8化と非同期NVENC保存。1088×1920の音声付き5秒動画を約60秒で生成 |
| [ComfyUI-MiniMax-H3-W4A4-VSA](https://github.com/sepiablue-ai/ComfyUI-MiniMax-H3-W4A4-VSA) | Ref2VAの高速化（FC1 W4A4、Streaming VSA） |
| [ComfyUI-H3-LBH-X2-Stream](https://github.com/sepiablue-ai/ComfyUI-H3-LBH-X2-Stream) | 低解像度3step＋高解像度1step。14秒動画で約25%短縮 |
| [ComfyUI-MiniMax-H3-PDMD-X2](https://github.com/sepiablue-ai/ComfyUI-MiniMax-H3-PDMD-X2) | PDMD 2step融合モデル。1088×1920の5秒動画を約44〜50秒 |

ライセンスは [GPL-3.0-only](LICENSE)。外部プロジェクトは別途取得し、元の配布内容・ライセンスを維持します。[出典](THIRD_PARTY_NOTICES.md)を参照してください。
