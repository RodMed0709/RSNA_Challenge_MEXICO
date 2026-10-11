# exp034_raptor_convnext — bitácora

- 2026-10-11 22:50 UTC · worktree `RSNA_wt034` desde `origin/exp030_raptor_etiquetas_L2` (456c455). Claim creado.
- Benchmark de backbones en rsna-d2 (96 img 320², bf16, grad ckpt): CoAtNet-2 1.120 s/paso, 14.6 GiB · ConvNeXt-B 384
  0.527 s/paso, 6.2 GiB · ConvNeXt-S 384 0.367 s/paso, 4.6 GiB → ConvNeXt-B (ver SPEC). Pesos HF Apache-2.0, 354 MB.
- Predicciones val/gold de exp030 (mejor época y SWA) copiadas a `/workspace/users/rodrigo/preds_ref/exp030_raptor_etiquetas_L2/`
  (desde `wt_exp030/results`, metrics fold 0: pseudo 0.8795, gold 0.9123, ep12).
