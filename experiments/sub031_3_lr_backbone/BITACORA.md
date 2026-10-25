# sub031_3_lr_backbone — bitácora

- 2026-10-12 · worktree `worktrees/RSNA_wt031_3` desde `origin/exp031_raptor_etiquetas_consenso` (ca4f9b3). Claim creado.
- Fold 0 de exp031 (`metrics_f0_backup.json`: pseudo 0.8836, gold 0.9144, ep15) y sus predicciones copiados a
  `/workspace/users/rodrigo/preds_ref/exp031_raptor_etiquetas_consenso/` (no se toca el results de rsna-f).
- Smoke omitido (solo cambia `BB_LR` en la celda 0).
- 00:11 UTC run real fold 0 en rsna-d2 (PID papermill 376358, log `results/sub031_3_lr_backbone/run_20261012_001157.log`
  en wt_sub031_3 del pod). `results/exp031_raptor_etiquetas_consenso/metrics.json` del clon (versionado en la rama) es
  idéntico byte a byte al backup fold 0 (pseudo 0.8836, gold 0.9144), así que el Δ de la celda 5 es contra el fold 0.
  train 3477 · val 871 · 434 pasos/época · VRAM ~18.5 GB.
  ep0 pseudo L2 0.7204 · v1 0.6933 · gold 0.7062 (exp031 ep0: 0.6955 / 0.6690 / 0.6832) · train 512 s + eval 72 s
  → 16 épocas ≈ 2.6 h (fin ≈ 02:50 UTC).
- Run terminado: mejor época 15 pseudo L2 0.8832 · v1 0.8372 · gold 0.9146; SWA [15,14,11] 0.8840 / 0.9148; 2.50 GPU-h → **didnt**. metrics.json copiado al repo.
