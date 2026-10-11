# sub031_1_consenso_jev

| Campo | Valor |
|---|---|
| Autor | rodrigo |
| Padre | exp031_raptor_etiquetas_consenso |
| Fecha | 2026-10-11 |
| Rama | sub031_1_consenso_jev |
| Estado | run (fold 0) |

Idea: sub-experimento de afinado de exp031 (consenso de 4 fuentes blandas, `worked`, pseudo L2 0.8836). Añadir una 5.ª
fuente al promedio: las etiquetas JEV de riadmohamed42 (`riadmohamed42/jev-knee-labels`, CC0).

## 1. SPEC (antes de tocar el notebook)

**Hipótesis:** añadir el ensemble JEV como 5.ª fuente al promedio por celda sube la pseudo AUC L2 en fold 0 en ≥ +0.003
contra exp031 sin que gold caiga más de 0.02.

**Única variable:** etiquetas de ENTRENAMIENTO = promedio por celda de 5 fuentes (padre: 4).

| Fuente | Archivo | Estado |
|---|---|---|
| Raptor | `labels/public/dreaddevelopment/rsna-knee-labels/labels_llm_soft.parquet` | como el padre |
| L2 | `labels/public/stevenleehans/rsna-knee-llm-report-labels/llm_labels_v4_blend.csv` | como el padre |
| L3 | `labels/public/pilkwang/rsna-knee-llm-labels/report_labels_v2.csv` | como el padre |
| L4 | `labels/public/lixin73/rsna-knee-llm-report-labels-sol56/labels_llm_gpt56sol.csv` | como el padre |
| **JEV** | `public_assets/riadmohamed42__jev-knee-labels/FINAL2_labels_scores.csv` | **nueva** |

**Qué es JEV y por qué este archivo.** El dataset trae dos tablas blandas:
- `FINAL2_labels_scores.csv` = el **ensemble de texto JEV** (6 lecturas del mismo modelo OpenJev con prompts v5–v9,
  `steps=4`, `think=512`, promediadas por hallazgo). Es la fuente JEV propiamente dicha y la que se usa aquí.
- `HYBRID_labels_scores.csv` = `0.95 × hard-swap(Steven 9 hallazgos / pilkwang ACL, MCL, Fractura) + 0.05 × ensemble`.
  Es ~95 % L2/L3, que ya están en el consenso: añadirla sería repesar L2/L3, no aportar una fuente nueva. La consigna
  era usar HYBRID solo si JEV no fuera blanda por celda o no cubriera el pool, y JEV sí cumple ambas cosas.

Comprobado en el pod antes del notebook: FINAL2 tiene 4407 estudios, 12 columnas, 0 NaN, 0 UID duplicados, valores en
[4.5e-7, 0.991] (52 884 valores distintos: blanda por celda), cubre 4348/4348 del pool. Media 0.148 (más baja que las
demás: Raptor 0.257, L2 0.338, L3 0.351, L4 0.260 en train). Acuerdo binarizado (> 0.5) en el pool: JEV–L2 0.879,
JEV–Raptor 0.909.

Mismo manejo que el padre: targets blandos tal cual, media de las fuentes presentes por celda, NaN enmascarado,
`pos_weight = clip((1−prev)/prev, 1, 10)` recalculado con la misma regla (consecuencia de la variable).

**Todo lo demás idéntico al padre:** corpus 96×384 en `/root/corpus96_384`, recorte central 320, CoAtNet-2 384 +
`RaptorClassifier`, 16 épocas, bs 8 × k 12 (k_eval 24), AdamW / OneCycle / LRs, bf16, semilla 42, `split_v1_seed42`
fold 0, selección por pseudo L2 (L2 binarizada > 0.5) en val fold 0, SWA top-3 por pseudo L2. Gold y v1 solo se reportan.

**Métrica principal:** pseudo L2 fold 0 de la mejor época. SWA top-3 se reporta.

**Criterios (fijos, no cambian después de correr):**
- `worked`: pseudo L2 fold 0 ≥ exp031 + 0.003 (= **≥ 0.8866**) **y** gold no cae > 0.02 contra exp031 (gold ≥ 0.8944).
  Umbral menor que el +0.005 habitual porque es un sub-experimento de afinado.
- `killed`: pseudo L2 fold 0 ≤ exp031 − 0.005 (= **≤ 0.8786**) **o** gold cae > 0.02 contra exp031.
- `didnt`: cualquier otro caso.
(Padre exp031, de `results/exp031_raptor_etiquetas_consenso/metrics.json`: pseudo L2 0.8836, gold 0.9144.)
La `decision_auto` del notebook usa el umbral estándar ±0.005; el verdict se aplica a mano con estos criterios.

**Sesgos declarados:**
- **Gold influyó en JEV en origen:** los pesos de cada lectura del ensemble se eligieron por su AUC en 67 estudios
  de gold-74, que incluye los 58 gold de la competición (7 reservados). Gold ya tocó los targets; como gold aquí solo
  veta, un gold alto en este experimento vale todavía menos como señal.
- La vara pseudo L2 está sesgada hacia etiquetas con L2 (ver exp031). Añadir JEV **diluye** el peso de L2+L4 en el
  promedio (de 2/4 a 2/5), así que el sesgo juega en contra de esta variante: una bajada leve de pseudo no implica
  peores etiquetas. El LB del modelo es el árbitro final.

**Presupuesto:** fold 0 ≈ 2.5 GPU-h (como el padre) en rsna-g (RTX PRO 4500 32 GB), corpus en `/root/corpus96_384`.

**Riesgos:** la media baja de JEV baja la escala del objetivo (y sube `pos_weight`); sesgos de arriba.

## 2. PLAN

- Notebook `sub031_1_consenso_jev.ipynb` = copia del de exp031; cambios: celda 0 (`EXP`, `PADRE`, `JEV_CSV`, `NOTES`),
  celda 2 (`SOURCES` + JEV; logs de nº de fuentes y acuerdo con JEV).
- Smoke: `SMOKE=True` (64 train / 48 val, 1 época) → `metrics_smoke.json`.
- Run real: `FOLDS=[0]`, `SMOKE=False`, en rsna-g.

## 3. VERDICT (números de `results/sub031_1_consenso_jev/metrics.json`)

**Resultado:** pendiente · **Fase alcanzada:** screen fold 0 en curso

| Métrica | Padre (exp031) | Este | Δ |
|---|---|---|---|
| pseudo L2 fold 0 (mejor época) | 0.8836 (ep 15) | | |
| pseudo L2 fold 0 (SWA top-3) | 0.8835 | | |
| v1 fold 0 | 0.8368 | | |
| gold (veto) | 0.9144 | | |
| GPU-h | 2.49 | | |

**Qué aprendimos:**

**Siguiente paso sugerido:**
