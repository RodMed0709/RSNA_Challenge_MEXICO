# exp044_raptor_ft_nartaa — bitácora

- 2026-10-12 02:08 UTC · pod rsna-c2 (corpus d96 en `/root/corpus96_384`, setup_exit=0, copy_done). Claim creado.
  Worktree desde `exp031_raptor_etiquetas_consenso`. Pesos de nartaa ya en
  `/workspace/data/public_assets/nartaa__rsna-knee-publication-swa-weights-20261007/` (no hizo falta descargarlos).
- Smoke (all-data, 64 estudios, 1 época): carga estricta OK, sha256 OK, `lab` en el mismo orden; gold-58 del modelo inicial
  (k_eval 24, sin TTA) = 0.9191 (metadatos de nartaa 0.9190). Pasa todas las celdas; `alldata/metrics_smoke.json`.
  Nota: el consenso cubre 4407 estudios (algunas fuentes incluyen gold); `fold_ids` excluye gold → train 4349.
- 02:10 UTC run real all-data (PID papermill 13644, lanzador `/root/run044.sh`, log
  `results/exp044_raptor_ft_nartaa/run_20261012_021036.log` en wt_exp044): 543 pasos/época, ~1.17 s/paso, VRAM ~19 GB.
  ep0 loss 0.7677 · gold 0.9197 (sanidad; sin olvido). 632 s/época → fin ≈ 03:15 UTC.
