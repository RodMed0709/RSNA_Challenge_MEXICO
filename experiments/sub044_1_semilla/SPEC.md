# sub044_1_semilla

| Campo | Valor |
|---|---|
| Autor | rodrigo |
| Padre | exp044_raptor_ft_nartaa (all-data: fine-tune 6 ép de los pesos SWA de nartaa con consenso exp031; gold SWA 0.9192) |
| Fecha | 2026-10-14 |
| Rama | sub044_1_semilla (desde exp044_raptor_ft_nartaa) |
| Estado | run (all-data) |

Idea: segundo miembro Raptor "init nartaa + nuestras etiquetas" con otra semilla, para promediar con exp044 dentro del
bloque Raptor del blend Kaggle (R7 + Raptor). Precedente: sub031_2 (seed 7) + exp031 promediados subieron pseudo +0.0036.

## 1. SPEC

**Hipótesis:** el promedio por rangos exp044 + sub044_1 es un Raptor "propio" más estable que exp044 solo; en el blend con
R7 no baja el LB y puede subirlo.

**Única variable cambiada:**

| Parámetro | Padre (exp044) | Este |
|---|---|---|
| `SEED` | 42 | 7 |

Todo lo demás idéntico (notebook de exp044 sin tocar, se ejecuta con `-p SEED 7`): init SWA nartaa (sha 7e5315da…), 6 épocas,
LR 1e-5/1e-4, SWA [3,4,5], all-data 4349, corpus d96/384. Contaminación igual que exp044 (sin validación limpia).

**Criterios (no cambian después de correr):**
- `killed` si gold SWA < 0.9192 − 0.02 = **0.8992** (veto).
- Decide el LB: `worked` si el kernel con Raptor = rank-mean(exp044, sub044_1) da LB ≥ el mismo kernel con exp044 solo + 0.002.
- Presupuesto ≤ 1.5 GPU-h.

## 2. PLAN

- Notebook de exp044 ejecutado con papermill `-p SEED 7 -p SMOKE False` (sin cambios de código; smoke omitido como en
  sub031_2). Pod `rsna-r44s` (RTX PRO 4500): copia del corpus a `/root/corpus96_384` → run → `swa_alldata.pt` →
  export a `/workspace/users/rodrigo/kaggle_export/sub044_1_semilla/` → podStop.

## 3. VERDICT

(pendiente)
