# exp043_raptor_3slots

| Campo | Valor |
|---|---|
| Autor | rodrigo |
| Padre | exp030_raptor_etiquetas_L2 (pseudo L2 f0 0.8795, gold 0.9123, mejor época 12; SWA [12,15,14] 0.8793 / 0.9115) |
| Fecha | 2026-10-11 |
| Rama | exp043_raptor_3slots |
| Estado | run (fold 0) |

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

**Resultado:** pendiente · **Fase alcanzada:** screen fold 0 en curso

| Métrica | Padre | Este | Δ |
|---|---|---|---|
| pseudo L2 fold 0 (mejor época, k 24) | 0.8795 | | |
| pseudo L2 fold 0 (SWA top-3) | 0.8793 | | |
| gold (veto) | 0.9123 | | |
| s/paso entrenamiento | ~1.19 | | |
| ventanas cobertura completa | 94 | 64 | −32 % |
| GPU-h | 2.50 | | |

**Qué aprendimos:**

**Siguiente paso sugerido:**
