# exp020_cabeza_epocas16

| Campo | Valor |
|---|---|
| Autor | rodrigo |
| Padre | exp013_cabeza_hallazgo |
| Fecha | 2026-10-08 |
| Rama | exp020_cabeza_epocas16 |
| Estado | killed |

Idea: exp013 (cabeza por hallazgo) con 16 épocas en lugar de 8.

Origen: exp015 mostró que la receta de 8 épocas está corta (exp002 → 16 épocas: +0.020 pseudo fold 0, meseta desde la
época 8, mejor la 12). exp013 mejoró la cabeza (+0.011). Antes de armar la referencia combinada hay que saber si las dos
ganancias se suman o se solapan.

## 1. SPEC (antes de tocar el notebook)

**Hipótesis:** si cambio `EPOCHS` de 8 a 16 en exp013, la pseudo AUC sube como en exp015, porque la cabeza por hallazgo
tiene más parámetros y también terminó su curva en la última época.

**Única variable cambiada:**

| Parámetro o celda | Padre (exp013) | Este experimento |
|---|---|---|
| `EPOCHS` | 8 | 16 |

Todo lo demás igual que exp013 (cache v1_256, 224 px, lr, semilla, etiquetas v1, cabeza, `SELECT_ON = "pseudo"`).
El scheduler coseno se estira a 16 épocas.

**Efecto esperado:** Δ pseudo fold 0 vs exp013 ≈ +0.005 a +0.015 (aditivo si ≥ +0.010).

**Criterios (no cambian después de correr):**
- Éxito → candidato para la referencia combinada: Δ pseudo fold 0 ≥ +0.005 contra exp013.
- Kill: Δ pseudo fold 0 ≤ −0.005, o gold < exp013 − 0.02 (fold 0 contra fold 0), o GPU-h > 1.5 × presupuesto.
- Neutral: `didnt` (las ganancias se solapan).
- Se reporta también la época del mejor checkpoint y las métricas de atención (entropía, coseno) frente a exp013.

**Presupuesto:** screen ~3.5 GPU-h (GPU compartida).

**Riesgos:** sobreajuste a etiquetas v1 ruidosas; se ve en `history_f0.csv`.

## 2. PLAN

- Copia del notebook de exp013. Celda que cambia: 0 (`EXP`, `PADRE`, `EPOCHS`, `NOTES`). Ninguna otra.
- Smoke con `SMOKE = True` antes del run real.

## 3. VERDICT (números de `results/exp020_cabeza_epocas16/metrics.json`)

**Resultado:** killed por presupuesto (GPU-h 1.503× > 1.5×), con AUC promising · **Fase alcanzada:** screen

| Métrica | Padre (exp013) | Este | Δ |
|---|---|---|---|
| pseudo AUC fold 0 | 0.838 | 0.850 | +0.012 |
| gold AUC fold 0 | 0.789 | 0.802 | +0.013 |
| Época del mejor checkpoint | 8 de 8 | 12 de 16 | |
| Entropía normalizada media (val) | ~0.97 | 0.958 (coseno 0.908) | |
| GPU-h | | 5.26 (316 min de reloj, GPU compartida por 3 runs) | 1.503× presupuesto (3.5) |

**Qué aprendimos (2-4 líneas):** las ganancias de cabeza (exp013) y épocas (exp015) se suman: +0.012 pseudo y +0.013 gold
sobre exp013, misma forma de curva que exp015 (meseta desde la época 8, mejor la 12, leve caída al final). El criterio de
presupuesto lo mata por 0.003× (5.26 GPU-h contra 5.25 de tope): el GPU-h se mide en reloj y la GPU estuvo compartida por 3 runs,
así que no refleja el costo real, pero el criterio no se cambia después de ver resultados.

**Siguiente paso sugerido:** el resultado de AUC justifica usar 12 épocas (no 16: la mejor es la 12 y ahorra 25 %) en la
referencia nueva. Para presupuestos futuros con GPU compartida, fijar el criterio en GPU-h efectivas o en épocas, no en reloj.
