# Bitácora — exp019_poda_slots

Padre: `exp012_entrada_v2` · Rama: `exp019_poda_slots` · Fase SDD actual: **IMPLEMENT, falta RUN**.

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

## 3. Cómo correrlo

```bash
cd /workspace/users/rodrigo/wt_exp019
RSNA_USER=rodrigo bash run_notebook.sh experiments/exp019_poda_slots/exp019_poda_slots.ipynb -p SMOKE False
```
Salidas en `results/exp019_poda_slots/`: `metrics.json`, `subsets.csv`, `per_finding_le2.csv`, `pareto_costo_auc.png`.

## 4. Resultados

Números solo desde `results/exp019_poda_slots/metrics.json`.
