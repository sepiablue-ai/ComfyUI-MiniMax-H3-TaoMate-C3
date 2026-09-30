# 採用した構成と結果 / Adopted configuration and evidence

C3は、TaoMate 3step・FL2VA I2V・FC1 W4A4・H3 sparse 10%に、C2 HeldMLP FFN8とC3 Held Attention Outを加えた構成です。C2/C3は重みを必要な範囲で保持し、繰り返しの転送やcastを削減します。全層をVRAMに常駐させる方式ではありません。C3はComfyUIの一時cast bufferを参照したまま保持せず、INT8データとscaleの独立コピーをattention呼び出し内で保持します。

The C3 wrapper runs after H3-Optimizations' final sampling-plan reconciliation and before ComfyUI places the model on the GPU. Changing this wrapper order can silently replace the optimization; the unmodified implementation explicitly verifies that order.

## 同一条件のC2→C3比較 / Controlled C2→C3 comparison

| 条件 / Condition | C2 seconds | C3 seconds | Saved |
|---|---:|---:|---:|
| cache-none, seed43 | 169.413 | 165.645 | 3.768 s |
| cache-none, seed44 | 169.567 | 164.916 | 4.651 s |
| normal cache, first | 171.766 | 167.282 | 4.484 s |
| normal cache, warm | 149.392 | 145.021 | 4.371 s / 2.926% |

採用時の比較動画は計10本。同seedのC2/C3で全復号映像・native PCMが一致しました。単発のwarm値からすべての解像度・prompt・GPUでの短縮率を保証しません。画像/音声の一致はC2とC3の差分に対する結果です。

Ten completed videos were used during adoption. Decoded video frames and native PCM were exact within same-seed pairs. These findings concern the C2→C3 holding change only. They do not establish losslessness relative to unquantized, dense, or full-step generation.

## 後日の確認 / Later observation

2026-09-28のVAE起動オプション比較では、通常C3のwarmは143.521秒、動画VAE nodeは30.108秒でした。`--fast fp16_accumulation` ONでは144.649秒、30.533秒となり、同seedの映像・音声も変わったため不採用です。このオプションをsetupやworkflowに含めていません。

The later 143.521-second observation came from a separate session. It must not be subtracted from the earlier C2 timing to claim a larger C3 speedup. The fp16 accumulation option was rejected in this configuration.

追加のカーネル実験、query chunk拡大、近似処理などの不採用案は配布対象に含めません。C3のコード・設定・入力は `source-provenance.json` とworkflowで固定しています。今回の配布用コードに対して動画を再生成したかどうかは、親フォルダの `VALIDATION.md` を参照してください。

Rejected kernel/approximation candidates are excluded. The package preserves the adopted C2/C3 source bytes. Packaging validation and its limits are recorded in `../VALIDATION.md`.
