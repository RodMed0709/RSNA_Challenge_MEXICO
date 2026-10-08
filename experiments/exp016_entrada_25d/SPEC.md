# exp016_entrada_25d

| Campo | Valor |
|---|---|
| Autor | rodrigo |
| Padre | exp002_dinov2s_ref |
| Fecha | 2026-10-08 |
| Rama | exp016_entrada_25d |
| Estado | spec |

Idea: entrada 2.5D: cada corte muestreado entra con sus dos vecinos como canales RGB, en lugar de repetir el mismo corte 3 veces.

Origen: `docs/08_LO_QUE_HACEN_OTROS.md` (los notebooks de ~0.94 usan ventanas de 3 cortes). Hoy los 3 canales del
backbone llevan la misma imagen: es capacidad desperdiciada. 2.5D da contexto 3D local sin costo extra de backbone.

## 1. SPEC (antes de tocar el notebook)

**Hipótesis:** si cada canal RGB lleva un corte distinto (i−1, i, i+1), la pseudo AUC sube, porque roturas de menisco y
ligamento se reconocen mejor con continuidad entre cortes vecinos.

**Única variable cambiada:**

| Parámetro o celda | Padre | Este experimento |
|---|---|---|
| Canales de cada imagen | corte i repetido ×3 | cortes (i−1, i, i+1); en los bordes se repite el corte del borde |

Todo lo demás igual: cache v1_256, 224 px, número de cortes muestreados, augmentations (se aplican igual a los 3 canales),
normalización ImageNet por canal, backbone, cabeza, épocas, lr, etiquetas v1, split, `SELECT_ON = "pseudo"`.

**Efecto esperado:** Δ pseudo fold 0 ≈ +0.005 a +0.02; costo ≈ igual (lectura de 3× cortes desde el cache en memoria).

**Criterios (no cambian después de correr):**
- Éxito → 5 folds: Δ pseudo fold 0 ≥ +0.005.
- Kill: Δ pseudo fold 0 ≤ −0.005, o gold < padre − 0.02, o GPU-h > 1.5 × presupuesto.
- Neutral: `didnt`.

**Presupuesto:** screen ~1.5–2 GPU-h (RTX PRO 4500 compartida: tiempo de reloj no comparable).

**Riesgos:** el preentrenamiento de DINOv2 espera canales correlacionados de una imagen natural; vecinos muy distintos
(espaciado grande) pueden confundir. Se reporta el espaciado medio entre cortes del cache.

## 2. PLAN

- Copia del notebook de exp002. Celda que cambia: 2 (datos: armado de los 3 canales). Ninguna otra.
- Smoke con `SMOKE = True` antes del run real.

## 3. VERDICT (números de `results/exp016_entrada_25d/metrics.json`)

**Resultado:** worked / didnt / killed · **Fase alcanzada:** screen / 5 folds

| Métrica | Padre | Este | Δ |
|---|---|---|---|
| pseudo AUC fold 0 | 0.827 | | |
| gold AUC (fold 0 contra fold 0 del padre) | 0.777 | | |
| GPU-h | | | |

**Qué aprendimos (2-4 líneas):**

**Siguiente paso sugerido:**
