# Task spec para Codex — stress_v1 (inferencia de robustez en pod)

Lee primero: `stress_test/SPEC_v1_modesto.md` (manda), `docs/now/CODEX.md`, `docs/now/BRIEF_agentes.md`, `docs/06_SDD.md`,
`experiments/exp040_robustez/{SPEC.md,infer_robustez.py,analisis_robustez.py,meta_subgrupos.py}`. Respeta CLAUDE.md/AGENTS.md del repo.

## Autorizado explícitamente por Rodrigo (21-oct)
- Crear UN pod GPU (`bash docs/now/runpod_pod.sh create rsna-stress "NVIDIA RTX PRO 4500 Blackwell"`), con tope de gasto 8 USD.
  Crear `guard_skip/<pod>` al empezar y quitarlo al terminar. Al final: apagar y BORRAR el pod. No dejar pods GPU encendidos.
- En el pod: `nuevo_experimento("stress_v1", padre="exp040")` para obtener el número global, y crear rama `expNNN_stress_v1`
  desde `main` (excepción a la regla de no crear ramas). NO push, NO merge. Commits locales en el pod; trae un git bundle.
- Reutilizar los scripts de exp040 parametrizados por checkpoint (declararlos en el PLAN de la SPEC del experimento). No tocar `src/knee/`, ni notebooks de otros, ni `split_v1_seed42`.

## Pasos
1. Auditoría (CPU, rsna-cron o pod): lista final del Panel R (Raptor fold 0 sin fuga; cada uno con ruta de pesos, ruta OOF, pseudo AUC propia). Escríbela en la SPEC del experimento ANTES de correr. Excluye lo que dice SPEC_v1 §1.
2. Smoke (20 estudios, 3 condiciones, 1 checkpoint). Gate: ref reproduce pseudo AUC (|Δ|<0.002).
3. Run: 16 condiciones de SPEC_v1 §2 por checkpoint del Panel R (fold 0) y por fold 0-4 de exp045 y exp054 (Panel F). Cadena autónoma con nohup, `status.log` y `HANDOFF.md` en `/workspace/users/rodrigo/orq/stress/`. Vigila el gasto; si proyecta > 8 USD, para y reporta.
4. Análisis: métricas de SPEC_v1 §3, H1–H3, tabla de LB (sin correlación), hard cases con Δ pareado. `metrics.json` agregado (celdas n ≥ 20, sin UIDs). Rellena la sección 3 (verdict) de la SPEC del experimento y la fila de `docs/EXPERIMENTOS.md`. Copia los números de `metrics.json`.
5. Trae a local solo: SPEC/bitácora/scripts/`metrics.json`, figuras agregadas. Nada de preds/UIDs/pesos.

## Addendum 21-oct (Panel S) — manda sobre lo anterior
Rodrigo quiere medir los ~20 envíos más recientes. Implementa también el Panel S de `SPEC_v1_modesto.md` (estabilidad sin etiquetas, `in_sample` marcado).
Tope de gasto 14 USD, hasta 2 pods en paralelo, plazo 22-oct 12:00 UTC. Orden de ejecución: auditoría → smoke → Panel S (tanques y 5 envíos más recientes primero) → Panel R → Panel F.
Haz checkpoints parciales en `metrics.json` por bloque para que haya algo útil aunque no termine todo. No cargues pesos de heliosli/nartaa a Git.

## Criterio de aceptación
Smoke y gate de reproducción pasan; todas las pasadas completan o se declara cuáles faltan y por qué; pod borrado; gasto real reportado;
mensaje final con Hecho / Verificación (salidas reales) / Pendiente. No cambies criterios tras ver resultados.
