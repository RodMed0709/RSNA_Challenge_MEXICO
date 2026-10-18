# exp080_seqhead_ctrl_mil

| Campo | Valor |
|---|---|
| Autor | rodrigo |
| Padre | exp074_raptor224_distill |
| Fecha | 2026-10-17 |
| Rama | exp080_seqhead_ctrl_mil |
| Estado | didnt (fold 0) · sonda LB didnt (0.949) |

Idea: el control de exp077/exp078 como experimento propio: cabeza original (attention-MIL) reentrenada sobre embeddings
del backbone exp074 congelado, evaluada con 64 ventanas. Sonda de LB por **decisión de Rodrigo del 17-oct**.

## 1. SPEC (antes de correr y de enviar)

**Hipótesis:** reentrenar solo la cabeza del padre sobre embeddings fijos de las 64 ventanas que usa el kernel (en vez
de entrenarla junto al backbone con 12 ventanas al azar y elegir épocas con 24) alinea entrenamiento e inferencia; en el
control de exp077/exp078 dio pseudo SWA f0 0.9018 (+0.0031) y gold 0.9252 (+0.0055).

**Única variable (contra exp074):** la cabeza se reentrena desde cero sobre el backbone `swa_f0.pt` de exp074 congelado,
con el protocolo de exp077/exp078 (64 ventanas ordenadas sin aug, hiperparámetros de cabeza del padre, SWA top-3 por
pseudo). Arquitectura de la cabeza igual a la del padre (`MILHead` = `RaptorClassifier.head`).

| Parámetro o celda | Padre (exp074) | Este experimento |
|---|---|---|
| Entrenamiento (celdas 3-4) | backbone + cabeza juntos, 12 ventanas al azar con aug | backbone congelado; solo la cabeza, 64 ventanas fijas |
| Evaluación | 24 ventanas | 64 ventanas (las del kernel) |

**Criterios (fijos antes de correr):**
- Fold 0 (regla normal; los números salen del `metrics.json` de este notebook, que repite el control de exp077/exp078,
  0.9018 / 0.9252): `promising` si Δ pseudo SWA f0 ≥ +0.005 (≥ 0.9037); `killed` si gold SWA f0 < 0.8997; si no `didnt`
  (neutral = didnt: no ahorra cómputo de inferencia). Con 0.9018 se espera `didnt`.
- Sonda LB (decisión de Rodrigo, independiente del fold 0): all-data = backbone all-data de exp074 congelado + cabeza MIL
  reentrenada en los 4349 no-gold, SWA sobre las épocas que elija la SWA de fold 0 de este notebook; kernel =
  `exp074-student224` con solo checkpoint y cabeza cambiados. `worked` en LB si ≥ **0.950** (padre LB 0.948).
- Runtime: el kernel padre ya usa K_EVAL = 64, así que el backbone hace el mismo trabajo; se mide la validación en Kaggle
  contra la de `exp074-student224` y se reporta (0.001 AUC ≈ 70 s en el premio de eficiencia).

**Presupuesto:** ≈ 0.2 GPU-h (embeddings fold 0 + all-data, compartidos con sub078_1). 1 envío.

**Riesgos:** la mejora de fold 0 (+0.003) está dentro de lo que mueve la semilla de cabeza (0.9013–0.9024) y la
selección de épocas usa el mismo val que se reporta; el LB puede no moverse. Fuga del teacher heredada de exp074.

## 2. PLAN
- Notebook = el de exp077 aplicado con `HEAD = "mil"` (celdas 0, 3, 4, 5, 6; `patch_nb.py`). `SMOKE` → `FOLDS = [0]` →
  `ALL_DATA = True` con las épocas SWA de fold 0 → `full_mil_alldata.pt` → dataset y kernel privado
  `exp080-ctrl224` → validación COMPLETE → `orq/seqhead/CANDIDATE_ctrl.json` y cola del 18-oct.

## 3. VERDICT (números de `results/exp080_seqhead_ctrl_mil/metrics.json` y `alldata/metrics.json`)

**Resultado fold 0:** didnt (regla normal) · **Sonda LB:** didnt — LB 0.949 < 0.950 (kernel `exp080-ctrl224` v1, ref 57330704, enviado 18-oct 00:07 UTC)

| Métrica | Padre exp074 | Este | Δ |
|---|---|---|---|
| pseudo AUC fold 0 (SWA ép [4, 2, 6]) | 0.8987 | 0.9018 | +0.0032 |
| gold AUC fold 0 (SWA) | 0.9197 | 0.9252 | +0.0055 |
| gold all-data (SWA ép [2, 4, 6], informativo) | — | 0.9242 | |
| GPU-h (fold 0 + all-data, embeddings incluidos) | 0.87 | 0.126 + 0.006 | |
| LB (criterio worked ≥ 0.950; padre 0.948) | 0.948 | 0.949 | +0.001 |

Repite exactamente el control de exp077/exp078 (0.9018 / 0.9252). Informativo: padre sin reentrenar con 64 ventanas
0.9004 / 0.9208; semillas 7 y 123: 0.9013 / 0.9212 y 0.9024 / 0.9242. Cabeza 0.22 ms/estudio (fp16); el backbone y las
64 ventanas son las mismas que en `exp074-student224`, así que la inferencia no cambia.

**Qué aprendimos (2-4 líneas):** Δ pseudo +0.0032 < +0.005 → `didnt` en fold 0. De ese +0.0032, +0.0017 sale solo de
evaluar con las 64 ventanas que ya usa el kernel y +0.0014 de reentrenar la cabeza sobre embeddings fijos; el kernel del
padre ya tenía las 64 ventanas, así que en LB solo puede notarse la segunda parte.

LB 0.949 vs padre 0.948 (+0.001, dentro del ruido de ~390 estudios): reentrenar la cabeza sobre embeddings fijos no se
nota en el LB público; no alcanza 0.950.

**Siguiente paso sugerido:** ninguno con esta cabeza; el alumno eficiente sigue siendo `exp074-student224(-fast)`.
