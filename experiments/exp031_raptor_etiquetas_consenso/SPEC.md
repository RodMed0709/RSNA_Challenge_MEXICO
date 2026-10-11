# exp031_raptor_etiquetas_consenso

| Campo | Valor |
|---|---|
| Autor | rodrigo |
| Padre | exp029_raptor_entreno |
| Fecha | 2026-10-11 |
| Rama | exp031_raptor_etiquetas_consenso |
| Estado | run (fold 0) |

Idea: en la receta Raptor/nartaa d96/384 de exp029, entrenar con un consenso de las 4 fuentes blandas públicas de
etiquetas (promedio por celda) en lugar de solo las de Raptor. Mismo espíritu que exp035 (promedio L2/L3/L4 sobre la
receta exp021), aquí sobre la receta Raptor y añadiendo la fuente del padre.

## 1. SPEC (antes de tocar el notebook)

**Hipótesis:** entrenar la receta de exp029 con el **promedio por celda de 4 fuentes blandas** sube la pseudo AUC L2 en
fold 0 en ≥ +0.005 sin que gold caiga más de 0.02.

**Única variable:** etiquetas de ENTRENAMIENTO. Padre: Raptor `labels_llm_soft` sola → este: media por celda de

| Fuente | Archivo (en `/workspace/data/labels/public/`) |
|---|---|
| Raptor (la del padre) | `dreaddevelopment/rsna-knee-labels/labels_llm_soft.parquet` |
| L2 | `stevenleehans/rsna-knee-llm-report-labels/llm_labels_v4_blend.csv` |
| L3 | `pilkwang/rsna-knee-llm-labels/report_labels_v2.csv` (columnas `<label>`; `__conf`/`__verdict` no se usan) |
| L4 | `lixin73/rsna-knee-llm-report-labels-sol56/labels_llm_gpt56sol.csv` |

Si una fuente no trae una celda (estudio × hallazgo), se promedian las presentes; celda sin ninguna fuente = NaN
enmascarado en la pérdida (media sobre celdas no-NaN; sin NaN es idéntica a la BCE media del padre). Mismo manejo que
el padre: targets blandos tal cual, `pos_weight = clip((1−prev)/prev, 1, 10)` con `prev = clip(nanmean de las
etiquetas de train, 0.03, 0.7)` (el `pos_weight` se recalcula con la misma regla sobre las nuevas etiquetas: es
consecuencia de la variable, no una segunda variable). Train = pool fold ≠ 0 ∩ estudios con etiqueta de consenso
(gold excluido como en el padre). Carga de L3/L4 = la de exp035 (`load_soft`, `pd.to_numeric`, chequeo [0, 1]).

**Todo lo demás idéntico al padre:** corpus `/root/corpus96_384` (96×384), recorte central 320, CoAtNet-2 384 +
`RaptorClassifier`, 16 épocas, bs 8 × k 12 (k_eval 24), AdamW / OneCycle / LRs, bf16, semilla 42, `split_v1_seed42`
fold 0, selección por pseudo L2 (L2 binarizada > 0.5) en val fold 0, SWA = media de las 3 mejores épocas por pseudo L2.
Gold y v1 solo se reportan.

**Métrica principal:** la del padre: pseudo L2 fold 0 de la **mejor época**. SWA top-3 se reporta.

**Criterios (fijos, no cambian después de correr):**
- `worked`: pseudo L2 fold 0 ≥ pseudo padre + 0.005 **y** gold no cae > 0.02 contra el padre.
- `killed`: pseudo L2 fold 0 ≤ pseudo padre − 0.005 **o** gold cae > 0.02 contra el padre.
- `didnt`: cualquier otro caso.

**Sesgos declarados:**
- **L2 y L4 coinciden al 99 %** (binarizado > 0.5 en el pool: 99.2 %, medido en exp035; L2 v4_blend ya incluye
  GPT-5.6-Sol = L4). El consenso pesa de hecho ~1/2 a GPT-5.6-Sol; las fuentes realmente distintas son Raptor y L3.
- **La vara pseudo L2 está sesgada hacia etiquetas que contienen L2**: este consenso incluye L2 (y su casi-copia L4) y
  el padre no, así que una subida de pseudo puede ser solo mayor acuerdo con L2. El LB del modelo fold 0 (envío
  aparte) es el árbitro final entre exp029, exp030 y exp031.

**Presupuesto:** fold 0 ≈ 2.5–3 GPU-h (como el padre) en rsna-f (RTX PRO 4500 32 GB), corpus en `/root/corpus96_384`.

**Riesgos:** sesgo de la vara (arriba); L3 tiene 5 niveles y L4 media más baja que L2: el promedio cambia la escala de
los objetivos (y por tanto `pos_weight`); el padre aún corre, la comparación se completa cuando termine.

## 2. PLAN

- Notebook `exp031_raptor_etiquetas_consenso.ipynb` = copia del de exp029; cambios: celda 0 (`EXP`, `PADRE`, `NOTES`,
  `L3_CSV`, `L4_CSV`), celda 2 (consenso por celda de las 4 fuentes, guarda de NaN, log de cobertura, nº de fuentes por
  celda, acuerdo binarizado entre fuentes y media del objetivo), celda 4 (pérdida con NaN enmascarado y `nanmean` para
  `prev`, como exp030; idéntica al padre sin NaN).
- Smoke: `SMOKE=True` (64 train / 48 val, 1 época) → `metrics_smoke.json`.
- Run real: `FOLDS=[0]`, `SMOKE=False`, en rsna-f.

## 3. VERDICT (números de `results/exp031_raptor_etiquetas_consenso/metrics.json`)

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
