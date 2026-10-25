# exp032_raptor_alineado_LR — bitácora

- 2026-10-11 19:05 UTC · worktree `worktrees/RSNA_wt032` desde `exp029_raptor_entreno` (a5dff96). Claim creado.
- `lateralidad_corpus.py` en rsna-d2 (57 s, 24 hilos): 20,792 series del corpus, 0 sin IOP. Eje x del paciente constante
  al 100 % por slot (sagital: orden de cortes, sentido −x; coronal/axial: columnas, sentido +x) → alinear = espejar las
  `L`. Lados: R 2,305 · L 2,041 · unknown 61. Tabla en `/workspace/data/cache/raptor96_384/align_lr_exp032.{npy,csv,json}`
  (copia en `/root/corpus96_384/`; `ids.npy` idénticos).
- `verifica_espejo.py` (15 s): correlación con la plantilla de 300 derechas, 20 izquierdas nativo → alineado:
  Sag fluid 0.537 → 0.561 (18/20) · Cor fluid 0.613 → 0.651 (19/20) · Ax 0.579 → 0.608 (19/20); control 20 R:
  0.575 / 0.655 / 0.624. 61 `unknown`, 0 espejados. Figura revisada a mano.
- Prueba de `StudyWindows` (CPU, 5 L + 2 R + 2 unknown): las ventanas del notebook son idénticas byte a byte a las de
  `align()` sobre el volumen completo; R y unknown idénticas a las nativas, L distintas.
- Smoke en rsna-d2 junto a exp029 (`SMOKE=True`, `BATCH_SIZE=2`, `N_WORKERS=4`, corpus completo `/root/corpus96_384`):
  pasa todas las celdas; 64 train / 48 val, 32 pasos en 38 s + eval 22 s; metrics.json OK (movido a `results/.../smoke/`).
- 19:14 UTC `/root/exp032/launch_real.sh` (setsid nohup, `RSNA_USER=rodrigo`) espera a que muera el PID 360626 de exp029
  y lanza `run_notebook.sh ... -p SMOKE False -y 'FOLDS: [0]'`. Log del lanzador: `/root/exp032/launch_real.log`.
  `results/exp029_raptor_entreno` enlazado en el clon para los Δ.
- exp029 terminó 20:11 UTC: mejor época 15 pseudo L2 **0.8682**, gold 0.9115; SWA [15,14,13] 0.8679 / 0.9111 (su
  metrics.json). Umbrales fijos de exp032: worked ≥ 0.8732 (gold ≥ 0.8915); killed ≤ 0.8632 o gold < 0.8915.
- 20:12 UTC run real fold 0 (PID papermill 367818, log `results/exp032_raptor_alineado_LR/run_20261011_201202.log` en
  wt_exp032 del pod): train 3477 · val 871 · 434 pasos/época · VRAM ~19 GB.
  ep0 pseudo L2 0.7095 · v1 0.7033 · gold 0.7132 (exp029 ep0: 0.7040 / 0.6865 / 0.7033) · train 512 s + eval 72 s
  (padre 499 + 62 s) → 16 épocas ≈ 2.6 h (fin ≈ 22:50 UTC).
- 22:43 UTC run terminado: mejor época 13 pseudo L2 0.8704 · v1 0.8299 · gold 0.9143; SWA [13,15,14] 0.8703 / 0.9154;
  2.50 GPU-h. Δ pseudo +0.0022 → **didnt**. metrics.json copiado al repo.
