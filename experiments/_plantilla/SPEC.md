# expNNN_tag

| Campo | Valor |
|---|---|
| Autor | <autor> |
| Padre | <padre> |
| Fecha | <fecha> |
| Rama | expNNN_tag |
| Estado | spec |

Idea: <idea>

## 1. SPEC (antes de tocar el notebook)

**Hipótesis:** si cambio <X> de <valor del padre> a <valor nuevo>, la pseudo AUC sube porque <razón>.

**Única variable cambiada:**

| Parámetro o celda | Padre | Este experimento |
|---|---|---|
| | | |

Todo lo demás igual: split `split_v1_seed42`, semilla, épocas, etiquetas.

**Efecto esperado:** Δ pseudo fold 0 ≈ +0.0xx; costo × <k>.

**Criterios (no cambian después de correr):**
- Éxito → 5 folds: Δ pseudo fold 0 ≥ +0.005.
- Kill: Δ pseudo fold 0 ≤ −0.005, o gold < padre − 0.02, o GPU-h > 1.5 × presupuesto.
- Neutral: cuenta como `didnt`, salvo ahorro de cómputo ≥ 30 %.

**Presupuesto:** screen <h> GPU-h · 5 folds <h> GPU-h · GPU <tipo>.

**Riesgos:** <memoria, inestabilidad, tiempo de inferencia en Kaggle (límite 9 h)…>

## 2. PLAN

- Celdas que cambian: <0 (parámetros: …), 3 (modelo: …)>. Ninguna otra.
- Smoke con `SMOKE = True` antes del run real.

## 3. VERDICT (números de `results/expNNN_tag/metrics.json`)

**Resultado:** worked / didnt / killed · **Fase alcanzada:** screen / 5 folds

| Métrica | Padre | Este | Δ |
|---|---|---|---|
| pseudo AUC fold 0 | | | |
| gold AUC (screen: fold 0 contra fold 0 del padre) | | | |
| OOF pseudo 5 folds | | | |
| GPU-h | | | |

**Qué aprendimos (2-4 líneas):**

**Siguiente paso sugerido:**
