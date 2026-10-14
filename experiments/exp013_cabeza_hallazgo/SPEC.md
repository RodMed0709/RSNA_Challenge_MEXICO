# exp013_cabeza_hallazgo

| Campo | Valor |
|---|---|
| Autor | rodrigo |
| Padre | exp002_dinov2s_ref |
| Fecha | 2026-10-08 |
| Rama | exp013_cabeza_hallazgo |
| Estado | worked |

Idea: cabeza con una query por hallazgo (12 queries) que atiende sobre las series, en lugar de un pooling único + MLP.

Es la pieza de método del paper (`docs/10_PLAN_PAPER.md`): si cada hallazgo aprende qué serie mirar, después se pueden
podar series en inferencia (experimento aparte, sin reentrenar).

## 1. SPEC (antes de tocar el notebook)

**Hipótesis:** si cambio la agregación entre series de una query única + MLP compartido a 12 queries, una por hallazgo,
la pseudo AUC sube, porque cada hallazgo se ve mejor en un plano (LCA en sagital, LCM en coronal, derrame en sagital/axial)
y una sola representación del estudio mezcla esa información.

**Única variable cambiada:** la agregación entre series y la cabeza.

| Parámetro o celda | Padre | Este experimento |
|---|---|---|
| Pooling entre series | 1 query aprendida → vector del estudio | 12 queries aprendidas (una por hallazgo), misma `MultiheadAttention`, mismos heads |
| Cabeza | LayerNorm → Linear → GELU → Dropout → Linear(12) | LayerNorm → Linear → GELU → Dropout por query (pesos compartidos) → un producto punto por hallazgo (Linear(d,1) por etiqueta, implementado como pesos [12, d]) |

Igual que el padre: backbone, cache v1_256, 224 px, pooling entre cortes, embeddings de plano/secuencia, épocas, lr
(cabeza nueva usa el `LR` de cabeza del padre), semilla, split, etiquetas v1, `SELECT_ON = "pseudo"`.

Instrumentación (no cambia el modelo): se guarda en `oof_f0.csv`/`gold_f0.csv` (o un `attn_f0.npz` aparte) el peso de
atención medio de cada hallazgo sobre cada serie, con plano y secuencia, para el análisis y para la poda posterior.

**Efecto esperado:** Δ pseudo fold 0 ≈ +0.005 a +0.02; costo ≈ igual (la cabeza es despreciable frente al backbone).

**Criterios (no cambian después de correr):**
- Éxito → 5 folds: Δ pseudo fold 0 ≥ +0.005.
- Kill: Δ pseudo fold 0 ≤ −0.005, o gold < padre − 0.02 (fold 0 contra fold 0), o GPU-h > 1.5 × presupuesto.
- Neutral: `didnt`, pero si la atención por hallazgo es interpretable (plano esperado por hallazgo) se conserva como base
  para el experimento de poda, porque esa propiedad es el objetivo de método, no solo la AUC.

**Presupuesto:** screen ~1.5 GPU-h (RTX PRO 4500) · 5 folds ~8 GPU-h.

**Riesgos:**
- 12 queries pueden colapsar a la misma atención: se reporta la entropía y la similitud entre las 12 distribuciones.
- Inestabilidad del padre al descongelar (fold 1): no se ataca aquí.

## 2. PLAN

- Copia del notebook de exp002. Celdas que cambian: 3 (modelo: `LabelQueryPool` + cabeza por etiqueta) y 5
  (evaluación: guardar atención por hallazgo). Ninguna otra.
- Smoke con `SMOKE = True` antes del run real.

## 3. VERDICT (números de `results/exp013_cabeza_hallazgo/metrics.json`)

**Resultado:** worked · **Fase alcanzada:** 5 folds

| Métrica | Padre | Este | Δ |
|---|---|---|---|
| pseudo AUC fold 0 | 0.827 | 0.838 | +0.011 |
| gold AUC (fold 0 contra fold 0 del padre) | 0.777 | 0.789 (folds 1–4: 0.787; padre 5 folds 0.788) | +0.012 |
| OOF pseudo 5 folds | 0.802 | 0.813 | +0.011 |
| GPU-h | | 11.0 (1.73 fold 0 + 9.31 folds 1–4; reloj, GPU compartida) | |

**Qué aprendimos (2-4 líneas):** la cabeza con una query por hallazgo se confirma en 5 folds: OOF pseudo 0.813 (+0.011 sobre
exp002), gold sin caída. Por fold: 0.838 / 0.823 / 0.807 / 0.777 / 0.817 (fold 3 es el más difícil, igual que en exp002). La
atención por hallazgo es casi uniforme (entropía ~0.97): la ganancia viene de cabezas separadas por hallazgo, no de seleccionar
series. Sobre la entrada v2 no suma (exp018); con 16 épocas sí (exp020, +0.012).

**Siguiente paso sugerido:** válida como cabeza para v1; en la referencia v2 se mantiene la cabeza de exp012 salvo que un
experimento v2 + 12 épocas muestre lo contrario. Para poda por hallazgo: atención que concentre sin colapsar (exp017/exp022).
