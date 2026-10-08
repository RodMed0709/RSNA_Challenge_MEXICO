# exp020_cabeza_epocas16

| Campo | Valor |
|---|---|
| Autor | rodrigo |
| Padre | exp013_cabeza_hallazgo |
| Fecha | 2026-10-08 |
| Rama | exp020_cabeza_epocas16 |
| Estado | spec |

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

**Resultado:** worked / didnt / killed · **Fase alcanzada:** screen

| Métrica | Padre (exp013) | Este | Δ |
|---|---|---|---|
| pseudo AUC fold 0 | 0.838 | | |
| gold AUC fold 0 | 0.789 | | |
| Época del mejor checkpoint | 8 de 8 | | |
| Entropía normalizada media (val) | ~0.97 | | |
| GPU-h | | | |

**Qué aprendimos (2-4 líneas):**

**Siguiente paso sugerido:**
