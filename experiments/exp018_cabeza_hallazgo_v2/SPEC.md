# exp018_cabeza_hallazgo_v2

| Campo | Valor |
|---|---|
| Autor | rodrigo |
| Padre | exp012_entrada_v2 |
| Fecha | 2026-10-08 |
| Rama | exp018_cabeza_hallazgo_v2 |
| Estado | didnt |

Idea: la cabeza por hallazgo de exp013 (12 queries, una por hallazgo) sobre la entrada v2 de exp012 (cache v2_05mm,
336 px, 6 slots).

Origen: exp013 subió pseudo +0.011 en fold 0 sobre exp002 (entrada v1). Con la entrada v2 cada token del pooling entre
series es un slot fijo (tipo de secuencia), así que la atención por hallazgo se puede leer por slot: es la base para podar
series por hallazgo (método del paper, `docs/10_PLAN_PAPER.md`).

## 1. SPEC (antes de tocar el notebook)

**Hipótesis:** la cabeza por hallazgo sube la pseudo AUC sobre la entrada v2 igual que subió sobre v1 (exp013: +0.011 en
fold 0), porque cada hallazgo se ve mejor en ciertos slots y una sola representación del estudio mezcla esa información.

**Única variable cambiada:** la agregación entre series y la cabeza (la de exp013).

| Parámetro o celda | Padre (exp012) | Este experimento |
|---|---|---|
| Pooling entre series | 1 query aprendida → vector del estudio (`key_padding_mask` en slots faltantes) | `LabelQueryPool`: 12 queries (una por hallazgo), misma `MultiheadAttention`, mismos heads, misma máscara |
| Cabeza | LayerNorm → Linear → GELU → Dropout → Linear(12) | LayerNorm → Linear → GELU → Dropout por query (pesos compartidos) → producto punto por hallazgo (pesos [12, d]) |

Igual que el padre: backbone DINOv2-small, cache v2_05mm, 336 px, 6 slots, `slot_embedding`, pooling entre cortes,
16 cortes, 8 épocas, `FREEZE_EPOCHS = 1`, lr 1e-4 / backbone 1e-5, batch 1 × acumulación 8, semilla, split
`split_v1_seed42`, etiquetas v1, `SELECT_ON = "pseudo"`.

Instrumentación (no cambia el modelo): `attn_f{fold}.npz` con la atención de cada hallazgo sobre cada token, con el índice
de slot de cada token (`slot`, `slot_name`, `plane`), en val y gold. En `metrics.json` (`per_fold.<fold>.attention`):
`slot_mass` por hallazgo (masa media de atención por slot; slot faltante = 0), `slot_mass_present` (solo estudios con ese
slot), `slot_coverage`, `plane_mass`, `entropy_norm` y `mean_pairwise_cosine`. Alimenta la poda de series por hallazgo.

**Efecto esperado:** Δ pseudo fold 0 ≈ +0.005 a +0.015; costo ≈ igual al padre (la cabeza es despreciable).

**Criterios (no cambian después de correr):**
- Éxito → 5 folds: Δ pseudo fold 0 ≥ +0.005 contra exp012.
- Kill: Δ pseudo fold 0 ≤ −0.005, o gold < exp012 − 0.02 (fold 0 contra fold 0), o GPU-h > 1.5 × presupuesto.
- Neutral: `didnt`.

**Presupuesto:** screen ~4.5 GPU-h (GPU compartida con otros runs) · 5 folds ~22 GPU-h.

**Riesgos:**
- 12 queries colapsan a la misma atención (como en exp013: entropía normalizada ~0.97, coseno 0.95): se reporta, no se
  corrige aquí (eso es exp017).
- La GPU compartida infla las GPU-h medidas por reloj: se anota en la bitácora.
- El padre exp012 aún no tiene fold 0: la comparación se hace cuando exista su `metrics.json`.

## 2. PLAN

- Copia del notebook de exp012. Celdas que cambian: 0 (`EXP`, `PADRE = "exp012"`, `NOTES`), 3 (modelo:
  `LabelQueryPool` + `KneeV2LabelQueryModel` sobre `KneeV2Model`) y 5 (evaluación: atención por hallazgo agregada por
  slot y `attn_f{fold}.npz`; el `metrics.json` del padre se busca también en `/workspace/users/<autor>/results`).
  Ninguna otra.
- Smoke con `SMOKE = True` antes del run real; su `metrics.json` se renombra a `metrics_smoke.json`.

## 3. VERDICT (números de `results/exp018_cabeza_hallazgo_v2/metrics.json`)

**Resultado:** didnt · **Fase alcanzada:** screen

| Métrica | Padre (exp012) | Este | Δ |
|---|---|---|---|
| pseudo AUC fold 0 | 0.850 | 0.846 | −0.004 |
| gold AUC (fold 0 contra fold 0 del padre) | 0.784 | 0.777 | −0.007 |
| OOF pseudo 5 folds | — | — | |
| Entropía normalizada media (val) | — | 0.992 | |
| Similitud coseno media entre hallazgos (val) | — | 0.976 | |
| GPU-h | | 4.42 (265 min de reloj, GPU compartida con exp012/exp020/exp021) | 0.98× presupuesto |

`slot_mass` por hallazgo (val, fold 0):

| Hallazgo | slot1 SAG FS | slot2 COR FS | slot3 AX FS | slot4 SAG PD/T2 | slot5 COR T1 | slot6 SAG T1 |
|---|---|---|---|---|---|---|
| ACL | 0.17 | 0.19 | **0.28** | 0.11 | 0.14 | 0.10 |
| MCL | **0.19** | 0.19 | 0.19 | 0.14 | 0.16 | 0.13 |
| Medial Meniscus | 0.19 | **0.23** | 0.15 | 0.15 | 0.17 | 0.12 |
| Lateral Meniscus | 0.21 | **0.21** | 0.19 | 0.14 | 0.16 | 0.10 |
| Medial OA | 0.20 | **0.23** | 0.19 | 0.13 | 0.16 | 0.09 |
| Lateral OA | **0.21** | 0.20 | 0.20 | 0.13 | 0.16 | 0.09 |
| PF OA | 0.17 | 0.21 | **0.21** | 0.14 | 0.15 | 0.11 |
| Effusion | 0.25 | 0.18 | **0.25** | 0.09 | 0.15 | 0.08 |
| Synovitis | 0.20 | 0.20 | **0.26** | 0.11 | 0.15 | 0.08 |
| Baker's | 0.18 | 0.22 | **0.30** | 0.10 | 0.12 | 0.08 |
| Contusion | 0.20 | 0.18 | **0.31** | 0.09 | 0.14 | 0.08 |
| Fracture | 0.21 | 0.21 | **0.27** | 0.10 | 0.14 | 0.08 |

**Qué aprendimos (2-4 líneas):** sobre la entrada v2 la cabeza por hallazgo no suma (Δ −0.004 pseudo, −0.007 gold): la ganancia
de exp013 en v1 (+0.011) la absorbe la entrada v2, que ya separa las series por slot. La atención sigue casi uniforme (entropía
~0.99), pero el sesgo débil es clínicamente coherente: los slots con supresión grasa (1–3) reciben más masa y el sagital T1
(slot 6) menos; ACL, Baker's y Synovitis tiran al axial FS, meniscos y OA medial al coronal FS.

**Siguiente paso sugerido:** la referencia v2 sigue con la cabeza de exp012. El método de poda por hallazgo necesita una atención
que concentre sin colapsar (exp022) y entrenamiento con slot dropout (exp019 mostró que sin reentrenar no tolera la poda).
