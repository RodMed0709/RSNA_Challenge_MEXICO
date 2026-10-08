# exp016_entrada_25d

| Campo | Valor |
|---|---|
| Autor | rodrigo |
| Padre | exp002_dinov2s_ref |
| Fecha | 2026-10-08 |
| Rama | exp016_entrada_25d |
| Estado | didnt |

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

**Resultado:** didnt · **Fase alcanzada:** screen

| Métrica | Padre | Este | Δ |
|---|---|---|---|
| pseudo AUC fold 0 | 0.827 | 0.824 | −0.003 |
| gold AUC (fold 0 contra fold 0 del padre) | 0.777 | 0.778 | +0.001 |
| GPU-h | | 1.69 (102 min de reloj, GPU compartida con exp013 folds 1–4) | |

**Qué aprendimos (2-4 líneas):** meter los cortes vecinos como canales RGB no mueve la AUC (Δ pseudo −0.003, gold +0.001):
neutral según la regla. El mejor checkpoint sigue siendo la última época (8/8). Con el cache v1_256 el espaciado entre cortes
muestreados es grande y el contexto local que aporta 2.5D lo cubre ya la agregación entre cortes de la cabeza. Costo igual.

**Siguiente paso sugerido:** no promover. Si se retoma, probar 2.5D sobre el cache v2 (0.5 mm, más cortes) dentro de la
referencia combinada, no sobre v1.
