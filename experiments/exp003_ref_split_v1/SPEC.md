# exp003_ref_split_v1

| Campo | Valor |
|---|---|
| Autor | rodrigo |
| Padre | exp002 |
| Fecha | 2026-09-24 |
| Rama | exp003_ref_split_v1 |
| Estado | running |

Idea: fijar la referencia del equipo sobre el split canónico.

## 1. SPEC

**Hipótesis:** la receta de exp002 sobre `split_v1_seed42` da una referencia reproducible contra la cual todos comparan.
No busca mejorar la AUC.

**Cambios contra exp002** (excepción a "una variable": es infraestructura, no modelo):

| Qué | exp002 | exp003 |
|---|---|---|
| Split | folds generados dentro del run | `split_v1_seed42` |
| Semillas | solo en el split | `seed_everything` + generador del DataLoader |
| Código | `train.py` | entrenamiento en celdas del notebook |
| Pérdida | NaN si un batch no tiene etiquetas válidas | devuelve 0 en ese caso |

Hiperparámetros idénticos: 224 px, 5 series, 16 cortes, batch 2 × 4 acumulación, 8 épocas, 1 congelada,
lr 1e-4 / backbone 1e-5.

**Efecto esperado:** pseudo fold 0 ≈ 0.82–0.83, gold ≈ 0.77–0.79 (distinto fold y distinto pool).

**Criterios:** sin decisión contra padre (splits distintos). Se acepta como referencia si corre completo y gold ≥ 0.75.

**Presupuesto:** fold 0 ≈ 0.5 GPU-h · 5 folds ≈ 2.5 GPU-h · RTX PRO 6000.

## 2. PLAN
Notebook nuevo con las 8 celdas fijas. Smoke primero, luego fold 0, luego 5 folds.

## 3. VERDICT
Pendiente.
