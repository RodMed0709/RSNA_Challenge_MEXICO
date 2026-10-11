# exp032_raptor_alineado_LR — bitácora

- 2026-10-11 19:05 UTC · worktree `RSNA_wt032` desde `exp029_raptor_entreno` (a5dff96). Claim creado.
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
