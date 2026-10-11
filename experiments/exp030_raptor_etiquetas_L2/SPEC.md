# exp030_raptor_etiquetas_L2

| Campo | Valor |
|---|---|
| Autor | rodrigo |
| Padre | exp029_raptor_entreno |
| Fecha | 2026-10-11 |
| Rama | exp030_raptor_etiquetas_L2 |
| Estado | run (fold 0) |

Idea: en la receta Raptor/nartaa d96/384 de exp029, entrenar con las etiquetas L2 (las mismas que nos dieron el mejor
LB propio en exp021) en lugar de las etiquetas blandas públicas de Raptor.

## 1. SPEC (antes de tocar el notebook)

**Hipótesis:** entrenar la receta de exp029 con etiquetas **L2 blandas** (`stevenleehans/rsna-knee-llm-report-labels/
llm_labels_v4_blend.csv`) en lugar de `dreaddevelopment/rsna-knee-labels/labels_llm_soft.parquet` sube la pseudo AUC L2
en fold 0 en ≥ +0.005 sin que gold caiga más de 0.02.

**Única variable:** etiquetas de ENTRENAMIENTO (Raptor `labels_llm_soft` → L2 v4 blend). Mismo manejo que el padre:
targets blandos tal cual (sin binarizar), `pos_weight = clip((1−prev)/prev, 1, 10)` con `prev = clip(media de las
etiquetas de train, 0.03, 0.7)`, BCE con logits. NaN enmascarado en la pérdida (media sobre las celdas no-NaN; L2 no
tiene NaN en los 4407 estudios, así que es idéntico a la BCE media del padre) y `nanmean` para `prev`. Train = pool
fold ≠ 0 ∩ estudios con etiqueta L2 (gold excluido como en el padre).

**Todo lo demás idéntico al padre:** corpus `/root/corpus96_384` (96×384, mismo `build_corpus96.py`), recorte central 320,
CoAtNet-2 384 + `RaptorClassifier`, 16 épocas, bs 8 × k 12 (k_eval 24), AdamW / OneCycle / LRs, bf16, semilla 42, split
`split_v1_seed42` fold 0, selección de época por pseudo L2 (L2 binarizada > 0.5) en val fold 0, SWA = media de las 3
mejores épocas por pseudo L2. Gold y v1 solo se reportan.

**Métrica principal:** la del padre: pseudo L2 fold 0 de la **mejor época** (la SPEC de exp029: "decide la mejor época").
SWA top-3 se reporta.

**Criterios (fijos, no cambian después de correr):**
- `worked`: pseudo L2 fold 0 (mejor época) ≥ pseudo padre + 0.005 **y** gold no cae > 0.02 contra el padre.
- `killed`: pseudo L2 fold 0 ≤ pseudo padre − 0.005 **o** gold cae > 0.02 contra el padre.
- `didnt`: cualquier otro caso.

**Sesgo declarado:** la vara interna (pseudo AUC contra L2 binarizada) favorece a priori entrenar con L2: el modelo
aprende la misma fuente con la que se le mide. Una subida de pseudo puede ser solo acuerdo con L2. Por eso el LB del
modelo fold 0 se medirá después (envío aparte) y **es el árbitro final** entre exp029 y exp030.

**Presupuesto:** fold 0 ≈ 2.5 GPU-h (como el padre) en rsna-c2 (RTX PRO 4500 32 GB), corpus ya copiado a `/root/corpus96_384`.

**Riesgos:** sesgo de la vara (arriba); L2 tiene valores mínimos de 0.005–0.01 en lugar de 0 (suelo de los blandos), lo
que cambia la escala de los targets negativos; el padre aún corre, la comparación se completa cuando termine.

## 2. PLAN

- Notebook `exp030_raptor_etiquetas_L2.ipynb` = copia del de exp029; cambios: celda 0 (`EXP`, `PADRE`, `NOTES`,
  `LABELS_SOFT` → L2), celda 2 (comentarios, guarda de NaN y log de cobertura de L2 sobre el pool), celda 4 (pérdida con
  NaN enmascarado y `nanmean` para `prev`; idéntica al padre sin NaN).
- Smoke: `SMOKE=True` (64 train / 48 val, 1 época) → `metrics_smoke.json`.
- Run real: `FOLDS=[0]`, `SMOKE=False`, en rsna-c2.

## 3. VERDICT (números de `results/exp030_raptor_etiquetas_L2/metrics.json`)

**Resultado:** pendiente · **Fase alcanzada:** screen fold 0 en curso

| Métrica | Padre (exp029) | Este | Δ |
|---|---|---|---|
| pseudo L2 fold 0 (mejor época) | pendiente (exp029 en curso) | | |
| pseudo L2 fold 0 (SWA top-3) | pendiente (exp029 en curso) | | |
| v1 fold 0 | pendiente (exp029 en curso) | | |
| gold (veto) | pendiente (exp029 en curso) | | |
| GPU-h | pendiente (exp029 en curso) | | |
| LB modelo fold 0 (árbitro final) | pendiente | pendiente | |

**Qué aprendimos:**

**Siguiente paso sugerido:**
