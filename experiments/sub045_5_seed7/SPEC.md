# sub045_5_seed7

| Campo | Valor |
|---|---|
| Autor | rodrigo |
| Padre | exp045_raptor_aug_espacial |
| Fecha | 2026-10-13 |
| Rama | sub045_5_seed7 (desde exp045_raptor_aug_espacial 607644c) |
| Estado | spec |

Idea: Fase 1 del estudio de preprocesado, P0: segunda semilla del padre (exp045 con semilla 7) para tener una referencia
de 2 semillas y medir el ruido de semilla antes de juzgar los brazos P1–P3 (exp058–exp060 aprox.).

## 1. SPEC (antes de tocar el notebook)

**Hipótesis:** exp045 con otra semilla da una pseudo L2 fold 0 dentro de ±0.01 de la de semilla 42 (0.8899); la media de
las dos es una referencia más estable que una sola corrida.

**Única variable cambiada:**

| Parámetro o celda | Padre | Este experimento |
|---|---|---|
| `SEED` (celda 0) | 42 | 7 |

Todo lo demás igual a exp045 (corpus `/root/corpus96_384`, split, épocas, etiquetas, aug, SWA, fold 0). Sin código nuevo.

**Efecto esperado:** Δ ≈ 0 ± 0.005.

**Criterios (fijos antes de correr):**
- `worked` (réplica válida, define la referencia): termina y |Δ pseudo fold 0 vs semilla 42| ≤ 0.01 y gold ≥ 0.9202 − 0.02.
- `didnt` (semilla inestable): |Δ pseudo| > 0.01 → la referencia sigue siendo la media, pero se avisa en ESTADO.
- `killed`: gold < 0.9202 − 0.02.
- Referencia del estudio = media(pseudo mejor época s7, 0.8898899); ref gold = media(gold s7, 0.9201799).

**Presupuesto:** smoke + fold 0 ≈ 2.7 GPU-h.

**Riesgos:** ninguno nuevo (misma receta).

## 2. PLAN

- Celda 0: `SEED = 7`, `NOTES`. Ninguna otra.
- Smoke y fold 0 con papermill, cola `orq/p7c_prepro/worker.sh` (pod A).

## 3. VERDICT (números de `results/sub045_5_seed7/metrics.json`)

Pendiente.
