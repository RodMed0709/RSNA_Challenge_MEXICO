# exp031_raptor_etiquetas_consenso — bitácora

- 2026-10-11 · rsna-f (RTX PRO 4500 32 GB, corpus en `/root/corpus96_384`). Claim creado. Padre exp029 corriendo en rsna-d2.
- Pod: worktree `wt_exp031_raptor_etiquetas_consenso`, `export RSNA_USER=rodrigo`, symlink `results/exp029_raptor_entreno` → `wt_exp029/results/...`; lanzador `/root/run031.sh` con `setsid nohup`.
- Smoke (corpus completo, 64/48, 1 época): pasa todas las celdas → `metrics_smoke.json`. Consenso: 4407 estudios, 0 celdas NaN; train fold 0: 41 724/41 724 celdas con las 4 fuentes. Acuerdo binarizado en train: L2–L4 0.992 · L2–L3 0.954 · L3–L4 0.950 · Raptor–L2 0.870 · Raptor–L3 0.847 · Raptor–L4 0.874. Media del objetivo: consenso 0.301 · Raptor 0.257 · L2 0.338 · L3 0.351 · L4 0.260.
- 19:31 UTC run real fold 0 (log `results/exp031_raptor_etiquetas_consenso/run_20261011_193113.log`): ~496 s train + ~61 s eval por época; ep0 pseudo L2 0.6955 · v1 0.6690 · gold 0.6832.
- Fin: mejor época 15, pseudo L2 0.8836, gold 0.9144, 2.49 GPU-h → `worked`. El metrics.json del fold 0 se guardó en el pod como `metrics_f0_backup.json` (el run all-data posterior reescribe la carpeta).
