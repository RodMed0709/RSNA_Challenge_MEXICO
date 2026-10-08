# Bitácora — exp019_poda_slots

Padre: `exp012_entrada_v2` · Rama: `exp019_poda_slots` · Fase SDD actual: **RUN en curso (screen fold 0)**.

## 1. Qué es

Experimento solo de inferencia. Se carga el checkpoint de exp012 fold 0 y se mide la pseudo/gold AUC y el costo
estimado en Kaggle usando solo algunos slots (el resto enmascarado con `key_padding_mask`, como una serie que falta).
Objetivo: track de eficiencia y la sección de poda del paper.

## 2. Registro

### 2026-10-08 — SPEC y notebook
- SPEC (secciones 1 y 2) commiteada antes del notebook. Número reservado en `/workspace/users/.claims/exp019_poda_slots`.
- Notebook: dataset y modelo copiados de exp012 sin cambios. La celda 4 extrae una vez por estudio y slot la salida de
  `backbone + slice_pool` (y mide el tiempo GPU por serie); cada subconjunto se evalúa solo con
  `slot_embedding + series_pool + head`. Se comprueba contra el forward completo (assert |Δprob| < 0.02).
- Costo de preprocesado: tabla por slot de `costo_inferencia.md` (s/serie v2, 1 hilo) escalada a 281 s para 6 slots;
  se imprime la comprobación contra los 182 s medidos para slots 1–3.

### 2026-10-08 — Smoke y run
- Smoke 1 falló en el assert del atajo (|Δprob| 0.63): bug `out[rows] = f` escribía la feature en todos los slots.
  Arreglado (`4ee5773`). Smoke 2 (40 pseudo + 20 gold, checkpoint exp012 época 8): atajo vs forward completo
  |Δprob| máx 1.95e-03; pipeline completo y `metrics_smoke.json` en el pod.
- Comprobación de costo: preprocesado slots 1–3 estimado 175 s vs 182 s medidos (−4 %).
- Primer run real falló en la celda 5 cuando ningún candidato cumple (rama `else` usaba `base` sin columnas Δ);
  arreglado (`b6ab759`) y relanzado. La extracción tarda ~4.4 min; el setup ~5 min por el volumen de red.

## 3. Cómo correrlo

```bash
cd /workspace/users/rodrigo/wt_exp019
RSNA_USER=rodrigo bash run_notebook.sh experiments/exp019_poda_slots/exp019_poda_slots.ipynb -p SMOKE False
```
Salidas en `results/exp019_poda_slots/`: `metrics.json`, `subsets.csv`, `per_finding_le2.csv`, `pareto_costo_auc.png`.

## 4. Resultados

Números solo desde `results/exp019_poda_slots/metrics.json`.
