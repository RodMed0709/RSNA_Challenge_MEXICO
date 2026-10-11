# exp030_raptor_etiquetas_L2 — bitácora

- 2026-10-11 · rsna-c2 (RTX PRO 4500 32 GB, corpus en `/root/corpus96_384`). Claim creado. Padre exp029 corriendo en rsna-d2.
- L2 (`llm_labels_v4_blend.csv`): 4407 estudios, 0 NaN, valores 0.005–1.0 (incluye los 58 gold, que se excluyen del train).
- Pod: worktree `wt_exp030` en el volumen, `export RSNA_USER=rodrigo`, symlink `results/exp029_raptor_entreno` → `wt_exp029/results/...`
  (sin `metrics.json` del padre todavía → `decision_auto` saldrá `sin_padre` si el padre no ha terminado; el verdict se hace a mano).
  Lanzador `/root/run030.sh` con `setsid nohup`.
- Smoke (corpus completo, 64 train / 48 val, 1 época): pasa todas las celdas; L2 cubre 4348/4348 del pool, 0 NaN; renombrado a `metrics_smoke.json`.
- 18:08 UTC run real fold 0 (PID papermill 1070, log `results/exp030_raptor_etiquetas_L2/run_20261011_180807.log` en wt_exp030):
  train 3477 · val 871 · 434 pasos/época · ~1.19 s/paso · VRAM ~19 GB. pos_weight (L2)
  [3.38, 4.58, 1.16, 3.41, 1.43, 2.07, 1.04, 1.0, 1.94, 1.78, 2.85, 4.52].
  ep0 pseudo L2 0.7181 · v1 0.6902 · gold 0.7171 (padre ep0: 0.7040 / 0.6865 / 0.7033). 515 s train + 75 s eval por época
  → 16 épocas ≈ 2.6 h (fin ≈ 20:45 UTC).
