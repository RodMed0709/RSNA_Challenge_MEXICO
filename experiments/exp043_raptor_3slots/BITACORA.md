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
- Run fold 0 (PID papermill 2698): train 3477 · val 871 · 434 pasos/época · 1.17 s/paso (padre ~1.19) · VRAM ~19 GB.
  ep0 pseudo L2 0.7214 · v1 0.6964 · gold 0.7114 (padre ep0: 0.7181 / 0.6902 / 0.7171). 508 s train + 73 s eval por
  época → 16 épocas ≈ 2.6 h (fin ≈ 00:15 UTC).
- Fin del run (≈ 00:15 UTC): mejor época 12, pseudo L2 0.8785, gold 0.8856; SWA [12,10,15] 0.8800 / 0.8850; 2.47 GPU-h.
  Verdict `killed` por el veto gold (−0.027). Reevaluaciones a 48 ventanas en la SPEC. metrics.json copiado al repo.
