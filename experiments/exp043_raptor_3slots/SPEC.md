# exp043_raptor_3slots

| Campo | Valor |
|---|---|
| Autor | rodrigo |
| Padre | exp030_raptor_etiquetas_L2 (pseudo L2 f0 0.8795, gold 0.9123, mejor época 12; SWA [12,15,14] 0.8793 / 0.9115) |
| Fecha | 2026-10-11 |
| Rama | exp043_raptor_3slots |
| Estado | killed |

Idea: eficiencia por diseño. exp040 (análisis de robustez del modelo exp029) mostró que quitar el slot Axial (4) y el
Coronal no fluido (3) en inferencia no cambia la AUC (Δ ≈ 0). Si el modelo se entrena y evalúa solo con los 3 slots
informativos, la inferencia necesita 64 ventanas de cobertura completa en lugar de 94 (−32 %) y lee ~31 % menos cortes DICOM.

## 1. SPEC (antes de tocar el notebook)

**Hipótesis:** si el modelo de exp030 se entrena y evalúa solo con los slots 0–2 (Sag fluid 26 + Sag no-fluid 22 +
Cor fluid 18 = 66 cortes) en lugar de los 5 (96 cortes), la pseudo AUC L2 en fold 0 **no empeora más de 0.003**
porque los slots 3 y 4 no aportan señal (exp040), y el coste de inferencia baja ~32 % (ventanas) y de lectura ~31 % (cortes).

**Única variable cambiada:**

| Parámetro o celda | Padre (exp030) | Este experimento |
|---|---|---|
| Slots que ve el modelo (`SLOTS_USE`, celda 0; uso en celdas 2–4) | 0–4 (96 cortes) | 0–2 (66 cortes) |

Mismo corpus (`/root/corpus96_384`, d96/384): los slots 3 y 4 simplemente no se cargan ni entran en las ventanas
(los centros válidos y las ventanas 2.5D se calculan sobre el subvolumen de 66 cortes). Todo lo demás igual: etiquetas
de entrenamiento L2 blandas, split `split_v1_seed42` fold 0, semilla 42, 16 épocas, bs 8 × k=12, k_eval 24, LRs, OneCycle,
pos_weight, bf16, recorte 320, selección de época por pseudo L2 (> 0.5) en val fold 0, SWA top-3 por pseudo L2.
Gold solo veta.

**Efecto esperado:** Δ pseudo fold 0 ≈ 0 (±0.003). Entrenamiento: mismo nº de ventanas por estudio (k=12), así que
s/paso ≈ igual; el ahorro está en inferencia (cobertura completa 64 vs 94 ventanas) y lectura de DICOM.

**Criterios (no cambian después de correr; no inferioridad):**
- `worked`: pseudo L2 fold 0 (mejor época, k_eval 24) ≥ padre − 0.003 (≥ 0.8765) **y** gold no cae más de 0.02 (≥ 0.8923).
- `killed`: gold < padre − 0.02.
- Si no: `didnt` (los slots 3/4 sí aportan algo cuando se entrena con ellos).
- Se reportan también (informativo, no decide): SWA, v1, s/paso, ventanas por estudio, y una reevaluación final con
  48 ventanas (exp040: 48 > 24) del mejor/SWA de este experimento y del SWA del padre (5 slots y solo 3 slots).

**Presupuesto:** screen fold 0 ≈ 2.5 GPU-h (RTX PRO 4500, pod rsna-e2).

**Riesgos:** estudios sin Cor fluid o sin sagital pierden más contexto (antes caían en el Coronal no fluido / Axial);
el criterio de no inferioridad con un solo fold tiene ruido del orden de ±0.003–0.005.

## 2. PLAN

- Celda 0: `SLOTS_USE = [0, 1, 2]`, `SLOT_SIZES`, `K_EVAL_EXTRA = 48`, `PARENT_SWA` (informativo).
- Celda 2: `USE_IDX` (índices de los cortes de los slots usados).
- Celda 3: `StudyWindows` trabaja sobre el subvolumen `USE_IDX` (máscara y centros sobre 66 cortes).
- Celda 4: `predict`/`loader` aceptan `k` y `use_idx`; al final, reevaluación informativa a 48 ventanas.
- Celdas 5–6: criterio de no inferioridad y campos de eficiencia en `metrics.json`.
- Smoke `SMOKE=True`, luego run real `FOLDS=[0]`.

## 3. VERDICT (números de `results/exp043_raptor_3slots/metrics.json`)

**Resultado:** killed · **Fase alcanzada:** screen fold 0

| Métrica | Padre (exp030) | Este (exp043) | Δ |
|---|---|---|---|
| pseudo L2 fold 0 (mejor época, k 24) | 0.8795 (ép. 12) | 0.8785 (ép. 12) | −0.001 |
| pseudo L2 fold 0 (SWA top-3, k 24) | 0.8793 [12,15,14] | 0.8800 [12,10,15] | +0.001 |
| v1 fold 0 (mejor época) | 0.8340 | 0.8333 | −0.001 |
| gold (veto, mejor época) | 0.9123 | 0.8856 | **−0.027** |
| gold (SWA) | 0.9115 | 0.8850 | −0.027 |
| s/paso entrenamiento | ~1.19 | 1.17 (494 s/época) | ≈ 0 |
| ventanas por estudio train / eval | 12 / 24 | 12 / 24 | 0 |
| ventanas cobertura completa (inferencia) | 94 | 64 | −32 % |
| GPU-h fold 0 | 2.50 | 2.47 | −1 % |

Reevaluación informativa a 48 ventanas (no decide):

| Modelo | cortes | pseudo L2 | v1 | gold |
|---|---|---|---|---|
| exp043 mejor época | 66 | 0.8811 | 0.8354 | 0.8901 |
| exp043 SWA | 66 | 0.8820 | 0.8355 | 0.8901 |
| exp030 SWA, 5 slots | 96 | 0.8836 | 0.8347 | 0.9108 |
| exp030 SWA, solo 3 slots en inferencia | 66 | 0.8830 | 0.8359 | 0.9085 |

Gold por hallazgo (mejor época, exp043 vs exp030): las caídas se concentran en Effusion 0.853 vs 0.925, Fracture
0.851 vs 0.919, Synovitis 0.712 vs 0.773, Baker's 0.955 vs 0.986, Lateral OA 0.805 vs 0.836; sube Medial Meniscus
0.977 vs 0.948.

Criterio fijo: pseudo cumple la no inferioridad (Δ −0.001 ≥ −0.003) pero gold cae 0.027 > 0.02 → **killed**
(`noninferiority_worked = false`).

**Qué aprendimos:** quitar los slots Cor no fluido y Axial no cambia la pseudo L2 (val 871) pero baja gold 0.027,
con las mayores caídas en hallazgos que se ven bien en axial (derrame, Baker, sinovitis) y en fractura. Gold son 58
estudios y su IC es ancho (una diferencia de 0.027 puede ser ruido), pero la señal coincide con la anatomía: el slot
axial parece aportar en gold. Llamativo: el modelo de 5 slots evaluado solo con 3 slots conserva gold (0.9085 a 48
ventanas), así que lo que se pierde es entrenar sin esos cortes, no verlos en inferencia. La pseudo L2 (etiquetas de
informes) no capta esta diferencia.

**Siguiente paso sugerido:** para eficiencia, mantener el entrenamiento con 5 slots y recortar solo en inferencia
(exp030 con 3 slots: −32 % ventanas, gold −0.002 a 48 ventanas), o probar 4 slots (0,1,2,4: sin Cor no fluido)
conservando el axial. Evaluar con 48 ventanas.
