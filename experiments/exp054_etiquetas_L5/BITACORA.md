# exp054_etiquetas_L5 — bitácora

- 2026-10-11 · Claim `/workspace/users/.claims/exp054_etiquetas_L5`. Rama desde `exp031_raptor_etiquetas_consenso` (ca4f9b3).
- Fase 1 en Windows (CPU + API, sin GPU). Datos copiados del volumen al scratchpad local (no a git).
  Modelo `gpt-4.1-mini`, 12 hilos, caché jsonl por prompt. Piloto 20 informes (muestra semilla 54): US$0.0385 →
  proyección US$8.5. Prompts sin tocar tras el piloto; la caché del piloto se reutilizó en la corrida completa.
- Corrida completa: 13 161 llamadas en ~27 min, 0 errores, **US$8.78**. `--build` → `L5_20261011.csv` (4407 × 12,
  sin NaN) + las 3 lecturas sueltas. Subidos a `/workspace/data/labels/ours/` (md5 L5 d27ee620…).
- Validación (`validar_L5.py` → `L5_validacion.json`): macro AUC vs L2 0.993 · L3 0.990 · L4 0.992 · Raptor 0.957 ·
  consenso4 0.992 (mín. Contusion 0.975) → **sana**. Gold solo reporte: 0.889.
- Notebook de fase 2 = copia de exp031 con L5 como 5.ª fuente (mismo diff que sub031_1 con JEV). Sin smoke todavía:
  todos los pods ocupados o asignados (H → exp045, I → exp046). Pedido pod al orquestador.
- 2026-10-12 · rsna-d2 libre a las 06:29 UTC (aviso del orquestador). El sondeo local no lanzó (esperaba `copy_done` en
  `/root/setup.log`, d2 copió el corpus por otra vía); lanzado a mano. Cadena `/root/run054_chain.sh` (chequeo torch →
  smoke → fold 0). Primer intento: el chequeo usaba entrada 320 en la CoAtNet-384 (error mío del chequeo, no del torch);
  corregido a 384. torch 2.8.0+cu128, archs incluyen sm_120, RTX PRO 4500 cap (12, 0): OK.
- Smoke 06:31–06:32 OK: consenso 4407 estudios, 41 724/41 724 celdas de train con 5 fuentes. Acuerdo binarizado en train:
  L5–Raptor 0.931 · L5–L2 0.882 · L5–L3 0.850 · L5–L4 0.887. Media del objetivo: consenso 0.278 · L5 0.182.
- Fold 0 06:32–09:04 UTC: ~499 s train + ~62 s eval por época; mejor época 15, pseudo L2 0.8896, gold 0.9190, SWA
  0.8895 / 0.9177, 2.50 GPU-h → `worked`. Copia `metrics_f0_backup.json` en el pod.
- All-data no lanzado: a las 09:04:54 otro agente lanzó sub045_2 en rsna-d2. Pendiente de pod.
- 2026-10-12 · Decisión del orquestador: sin all-data propio; el consenso con L5 entra en el combo exp056 (5 folds), que
  sustituye al all-data. exp054 cerrado.
