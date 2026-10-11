# exp043_raptor_3slots — bitácora

- 2026-10-11 · pod rsna-e2 (112 vCPU, RTX PRO 4500 32 GB, corpus d96 copiado a `/root/corpus96_384`, setup_exit=0, copy_done).
  Claim creado. Worktree desde `exp030_raptor_etiquetas_L2`; notebook = el de exp030 con `SLOTS_USE = [0, 1, 2]`.
- Smoke (corpus completo, 64 train / 48 val, 1 época): 66 de 96 cortes (65.4 válidos de media), cobertura completa 64
  ventanas (padre 94). Dos fallos corregidos: (1) la reevaluación a 48 ventanas con 4 estudios por forward (192
  imágenes) rompe `canUse32BitIndexMath` → el lote de evaluación se escala para mantener 96 imágenes por forward;
  (2) comentario mal puesto (SyntaxError). Después pasa todas las celdas; renombrado a `metrics_smoke.json`.
- Gotcha pod: `git pull` en el worktree falla porque el filtro nbstripout marca el notebook como modificado → usar
  `git -c filter.nbstripout.clean=cat -c filter.nbstripout.smudge=cat merge --ff-only FETCH_HEAD`.
- 21:39 UTC run real fold 0 (lanzador `/root/run043.sh`, PID papermill en `results/exp043_raptor_3slots/pid`,
  log `results/exp043_raptor_3slots/run_20261011_213858.log` en wt_exp043).
