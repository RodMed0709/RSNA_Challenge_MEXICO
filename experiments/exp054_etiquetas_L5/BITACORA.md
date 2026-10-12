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
