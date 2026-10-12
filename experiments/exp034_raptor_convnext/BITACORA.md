# exp034_raptor_convnext — bitácora

- 2026-10-11 22:50 UTC · worktree `RSNA_wt034` desde `origin/exp030_raptor_etiquetas_L2` (456c455). Claim creado.
- Benchmark de backbones en rsna-d2 (96 img 320², bf16, grad ckpt): CoAtNet-2 1.120 s/paso, 14.6 GiB · ConvNeXt-B 384
  0.527 s/paso, 6.2 GiB · ConvNeXt-S 384 0.367 s/paso, 4.6 GiB → ConvNeXt-B (ver SPEC). Pesos HF Apache-2.0, 354 MB.
- Predicciones val/gold de exp030 (mejor época y SWA) copiadas a `/workspace/users/rodrigo/preds_ref/exp030_raptor_etiquetas_L2/`
  (desde `wt_exp030/results`, metrics fold 0: pseudo 0.8795, gold 0.9123, ep12).
- Smoke (`SMOKE=True`, bs 8, rsna-d2 libre): pasa todas las celdas, incluida la de diversidad (correlación y ensamble
  contra las predicciones de exp030); 8 pasos en 29 s + eval 11 s; metrics.json OK (movido a `results/.../smoke/`).
- 22:54 UTC run real fold 0 (PID papermill 372876, log `results/exp034_raptor_convnext/run_20261011_225441.log` en
  wt_exp034 del pod; `results/exp030_raptor_etiquetas_L2` enlazado para los Δ): train 3477 · val 871 · 434 pasos/época ·
  ~0.60 s/paso · VRAM 8.5 GB · GPU 100 %.
  ep0 pseudo L2 0.7311 · v1 0.7046 · gold 0.7274 (exp030 ep0: 0.7181 / 0.6902 / 0.7171) · train 259 s + eval 46 s
  → 16 épocas ≈ 1.4 h (fin ≈ 00:20 UTC).
- 00:08 UTC run terminado: mejor época 11 pseudo L2 0.8587 · v1 0.8209 · gold 0.8681; SWA [11,14,15] 0.8584 / 0.8692; Spearman con exp030 0.883; ensamble por rangos 0.8769 (−0.0026); 1.21 GPU-h → **didnt**. metrics.json copiado al repo.
