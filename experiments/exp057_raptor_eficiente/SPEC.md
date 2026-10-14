# exp057_raptor_eficiente

| Campo | Valor |
|---|---|
| Autor | rodrigo |
| Padre | exp045_raptor_aug_espacial |
| Fecha | 2026-10-12 |
| Rama | exp057_raptor_eficiente |
| Estado | worked (fold 0 + all-data hechos) |

Idea: producto para el premio de eficiencia (score = AUC relativo + RuntimeSeconds/32400; menos runtime es mejor).
exp045 (mejor receta Raptor propia) entrenado e inferido a menor coste: entrada 224, 3 slots, sin TTA.

## 1. SPEC (antes de tocar el notebook)

**Tipo de experimento: PRODUCTO / EFICIENCIA (excepción declarada a "una variable").** Rodrigo lo autorizó el
2026-10-12 ~15:10 UTC. Se agrupan a propósito tres cambios de coste porque el objetivo es medir el trade-off AUC vs
runtime del paquete que se enviaría al premio de eficiencia, no atribuir el efecto a cada cambio. Las piezas ya se
midieron por separado antes: 3 slots en inferencia con nuestros modelos ≈ 0 (exp043: 0.8830 vs 0.8836), TTA espejo
+0.0016 (exp040), 224 + half con pesos nartaa −0.0023 gold (exp041 Ch). Si interesa atribuir, serán sub-experimentos.

**Hipótesis:** exp045 entrenado e inferido con entrada 224 (recorte central 320 reescalado), solo slots 0–2 (Sag fluido,
Sag, Cor fluido) y sin TTA pierde ≤ 0.010 de pseudo L2 en fold 0 contra exp045 (0.8899) y es ≥ 2.5× más rápido en
inferencia por estudio, porque los hallazgos se leen casi todos en sagital y coronal fluido (exp040: quitar Ax y Cor
no-fluido ≈ 0) y 224 conserva el campo de visión de 140 mm.

**Cambios agrupados (todos de coste):**

| Parámetro o celda | Padre (exp045) | Este experimento |
|---|---|---|
| Resolución de entrada (train, val, inferencia) | recorte central 320 del corte 384 | recorte central 320 → `F.interpolate` bilineal a **224** (`align_corners=False`, igual que el kernel Ch de exp041), antes de normalizar; backbone con `img_size=224` |
| Slots / cortes usados (train, val, inferencia) | 5 slots, 96 cortes | **slots 0, 1, 2** (Sag fluido 26, Sag 22, Cor fluido 18 = cortes 0–65); los centros de ventana solo salen de esos cortes |
| TTA en inferencia (kernel Kaggle) | espejo L/R anatómico (2 pasadas) | **sin TTA** (la validación interna de exp045 ya era sin TTA) |

Todo lo demás idéntico al padre: aug espacial de exp045 (espejo + afín, en GPU, aplicada a 320 antes de reescalar),
etiquetas de consenso de 4 fuentes, corpus 96×384, `RaptorClassifier`, 16 épocas, bs 8 × K 12 (K_EVAL 24), AdamW /
OneCycle / LRs, bf16, semilla 42, `split_v1_seed42` fold 0, selección por pseudo L2 (L2 binarizada > 0.5), SWA top-3.

**Medida de runtime (fija):** celda 5, `bench()`: 64 estudios de val fold 0, mismo GPU, fp16 + channels_last, 1 estudio
por llamada, 5 de calentamiento; tiempo H2D + forward. Configuración padre = 94 ventanas densas sobre los 96 cortes a
320 con TTA espejo (2 pasadas); configuración exp057 = ventanas densas sobre slots 0–2 a 224 sin TTA.
`speedup = s_por_estudio(padre) / s_por_estudio(exp057)` → `metrics.json["bench_inference"]`. El preprocesado DICOM en
CPU (que también baja al leer 3 de 5 series) no se mide aquí; el runtime real lo da el envío a Kaggle (informativo).

**Métrica principal:** pseudo L2 fold 0 de la mejor época; SWA top-3 se reporta.

**Criterios (fijos, no cambian después de correr):** padre exp045 fold 0 (`results/exp045_raptor_aug_espacial/metrics.json`):
pseudo L2 0.8899 (ep 15), SWA 0.8895, gold 0.9202.
- `worked`: pseudo L2 fold 0 ≥ 0.8899 − 0.010 (= **≥ 0.8799**) **y** `speedup` ≥ **2.5** **y** gold ≥ 0.9202 − 0.02 (= **≥ 0.9002**).
- `killed`: gold < 0.9002, o pseudo L2 fold 0 < 0.8749 (pérdida > 0.015).
- `didnt`: cualquier otro caso (p. ej. pierde entre 0.010 y 0.015, o `speedup` < 2.5).

**All-data (pedido por Rodrigo para el LB, encadenado sin esperar el verdict):** `ALL_DATA True`, semilla 42,
`SWA_EPOCHS` = las 3 épocas del SWA top-3 de fold 0 (`metrics.json["per_fold"]["0"]["swa"]["epochs"]`, regla fijada aquí;
si falta, [13, 14, 15]). → `results/exp057_raptor_eficiente/swa_alldata.pt` (res 224). Se envía a Kaggle solo si el
verdict de fold 0 no es `killed`. Envío sugerido: exp057 all-data solo, 3 slots, sin TTA, half (premio de eficiencia).

**Presupuesto:** fold 0 ≈ 0.6–0.8 GPU-h y all-data ≈ 0.7–0.9 GPU-h en RTX 5090 (≈ 2× eso en RTX PRO 4500). Kill de
coste: > 1.5 × 2.51 GPU-h (exp045) en el fold 0.

**Riesgos:** CoAtNet-2 a 224 desde pesos 384 (pos. relativa MLP, tamaño libre; probado en CPU). Con menos ventanas
distintas por estudio el SWA puede ser algo más ruidoso. El kernel de Kaggle debe replicar exactamente recorte 320 →
224 bilineal y los 3 slots (lectura de 3 series, como `submit_C3h` de exp041).

## 2. PLAN

- Notebook `exp057_raptor_eficiente.ipynb` = copia del de exp045 (`nuevo_experimento`); cambios:
  - celda 0: `PADRE`, `RES_IN = 224`, `USE_SLOTS = [0, 1, 2]`, `BENCH_N = 64`, `NOTES`.
  - celda 3: `IN_SLOTS`; `StudyWindows` toma centros solo de cortes válidos en `USE_SLOTS`; `to_input` reescala a
    `RES_IN` tras la aug y antes de normalizar; `build_backbone(img_size=RES_IN)`.
  - celda 4: los checkpoints guardan `"res": RES_IN`.
  - celda 5: `bench()` de inferencia por estudio (arriba).
  - celda 6: `metrics.json["bench_inference"]`.
- Smoke (`SMOKE True`) en el pod; si falla, la cadena se detiene y lo marca.
- Run: fold 0 → all-data (seed 42) encadenados en el pod de GPU (cola compartida con exp054 folds 1–4, ver
  `/workspace/users/rodrigo/p7b/`).

## 3. VERDICT (números de `results/exp057_raptor_eficiente/metrics.json`)

**Verdict: `worked`** (escrito por el orquestador en la nube, 2026-10-12 21:10 UTC, con los criterios fijos de la sección 1).

| Medida | exp057 | Umbral worked | Padre exp045 |
|---|---|---|---|
| pseudo L2 fold 0, mejor época (ep 15) | **0.8892** | ≥ 0.8799 ✔ | 0.8899 |
| pseudo L2 SWA top-3 (ép. 15, 14, 13) | 0.8889 | (se reporta) | 0.8895 |
| gold (58) mejor época | **0.9092** | ≥ 0.9002 ✔ | 0.9202 |
| gold SWA | 0.9096 | — | — |
| v1 mejor época / SWA | 0.8452 / 0.8448 | — | — |
| `bench_inference.speedup` (64 estudios, RTX PRO 4500) | **6.64×** (0.3713 → 0.0560 s/estudio) | ≥ 2.5 ✔ | — |

- Δ pseudo L2 f0 −0.0007 (`delta.pseudo_f0`), Δ gold −0.0110 (`delta.gold`, dentro del veto de 0.02). 1.20 GPU-h fold 0.
- Lectura: a 224 px, 3 slots y sin TTA la pseudo L2 queda prácticamente igual que el padre con 6.6× menos tiempo de GPU
  por estudio. El gold baja 0.011 (Lateral OA 0.795, Synovitis 0.792 en gold), sin llegar al veto; vigilarlo en el LB.
- All-data (`alldata/metrics.json`): SWA = épocas [15, 14, 13] de fold 0 (regla de la SPEC), 1.34 GPU-h, bench 6.63×
  sobre 58 estudios. Pesos `results/exp057_raptor_eficiente/swa_alldata.pt`. Como el verdict no es `killed`, se envía a
  Kaggle como candidato al premio de eficiencia (kernel `exp057-eff-alldata`; el runtime real lo da el envío).
