# sub056_2_L5v2 — bitácora

- 2026-10-12 10:13 UTC · Tarea del orquestador (autorizada por Rodrigo). Claim `sub056_2_L5v2`. Rama desde exp056 (2083da0).
- Pilotos de 20 informes para D/E: gpt-5-mini → 404 (organización no verificada). gpt-5.4-mini `none` ~US$18.9
  proyectado para D+E, `low` ~US$24; gpt-5.4-nano `low` ~US$6.6. Sin caché de prompt (prefijo < 1024 tokens).
  Decisión por costo: D en gpt-5.4-mini `none`, E en gpt-5.4-nano `low`. Prompts sin tocar tras el piloto.
- D_strict 4407 informes, 0 errores, US$8.27 (~9 min). E_checklist 4407, 0 errores, US$3.46.
- `--build` → `L5v2_20261012.csv` (md5 e71d0fea…) + 5 lecturas sueltas en `/workspace/data/labels/ours/`. Validación:
  sana (macro AUC vs consenso4 0.994). Marcador `L5v2_READY` creado.
- rsna-d2 (asignado por el orquestador): worktree `wt_sub056_2_L5v2`, symlink `results/exp056_raptor_combo` → worktree
  de exp056; lanzador `/root/run056_2_chain.sh` (PID 396290) espera a que no haya papermill y exista `L5v2_READY`,
  luego smoke → fold 0.
- Smoke 11:41–11:43 OK (41 724/41 724 celdas con 5 fuentes; media del objetivo 0.276; L5v2–Raptor 0.926, –L2 0.873).
- Fold 0 11:43–14:15 UTC: pseudo L2 0.8874 (ep 15), gold 0.9201, 2.51 GPU-h; padre exp056 fold 0 0.8857 / 0.9198 →
  Δ +0.0017 → `didnt`. L5v3 no se lanza. rsna-d2 detenido (stop) al terminar.
