# sub031_2_semilla

| Campo | Valor |
|---|---|
| Autor | rodrigo |
| Padre | exp031_raptor_etiquetas_consenso |
| Fecha | 2026-10-12 |
| Rama | sub031_2_semilla |
| Estado | spec |

Idea: repetir exp031 (Raptor, etiquetas consenso de 4 fuentes, fold 0) con otra semilla para medir la varianza entre
corridas y obtener un segundo miembro para ensamble.

Origen: los deltas de exp030–exp033 sobre exp029 van de +0.001 a +0.015 en pseudo L2 con un solo fold y una sola semilla.
Sin medir el ruido de semilla no sabemos qué umbral de decisión (±0.003 / ±0.005) es razonable.

## 1. SPEC

**Hipótesis:** la diferencia de pseudo L2 entre semillas es < 0.003 (los umbrales de ±0.005 usados en exp030–exp033 son
mayores que el ruido) y el promedio por rangos de las dos semillas sube pseudo L2 ≥ +0.002 sobre exp031 solo.

**Única variable cambiada:**

| Parámetro | Padre (exp031) | Este |
|---|---|---|
| `SEED` | 42 | 7 |

Todo lo demás idéntico (corpus 96×384, fold 0, etiquetas consenso, 16 épocas, selección por pseudo L2, SWA top-3).

**Criterios (no cambian después de correr):**
- Se reporta |Δ pseudo L2| y |Δ gold| entre semillas como estimación del ruido.
- `worked` si |Δ pseudo L2| < 0.003 (ruido pequeño: los umbrales actuales son válidos) — esto es una medición, no una mejora.
- Además, informativo: pseudo L2 del promedio por rangos de exp031 + sub031_2 en val fold 0.

**Presupuesto:** ~2.5 GPU-h (rsna-c2).

## 2. PLAN

- Copia del notebook de exp031. Celda que cambia: 0 (`EXP`, `PADRE`, `SEED`). Ninguna otra.
- Smoke omitido: el notebook es idéntico al de exp031 (ya pasó smoke y run completo); solo cambia la semilla.

## 3. VERDICT (números de `results/sub031_2_semilla/metrics.json`)

**Resultado:** · **Fase alcanzada:** screen

| Métrica | exp031 (seed 42) | sub031_2 (seed 7) | Δ |
|---|---|---|---|
| pseudo L2 fold 0 | 0.8836 | | |
| gold | 0.9144 | | |
| pseudo L2 ensamble 2 semillas | — | | |

**Qué aprendimos:**
