# sub078_1_bigru_alldata

| Campo | Valor |
|---|---|
| Autor | rodrigo |
| Padre | exp078_seqhead_bigru (fold 0 `didnt`); padre LB = exp074-student224 (LB 0.948) |
| Fecha | 2026-10-17 |
| Rama | sub078_1_bigru_alldata |
| Estado | running (LB pendiente) |

Idea: sonda de LB de la cabeza Bi-GRU de exp078 en all-data. **Decisión de Rodrigo del 17-oct**: llevarla al LB aunque
el fold 0 salió `didnt` (pseudo SWA f0 0.9017 < 0.9037 y no > control 0.9018). No cambia el verdict de exp078.

## 1. SPEC (antes de correr y de enviar)

**Pregunta:** ¿la cabeza Bi-GRU sobre el backbone all-data de exp074 congelado mejora al alumno eficiente en LB?

**Única variable (contra el kernel padre `exp074-student224`):** la cabeza. Backbone = `swa_alldata.pt` de exp074
(`kaggle_export/exp074_raptor224_distill/`) congelado; cabeza Bi-GRU (2 × 256) + attention-MIL entrenada en los 4349
no-gold con la receta exacta de exp078 (embeddings de 64 ventanas ordenadas sin aug, AdamW 1e-3, wd 0.02, OneCycle 0.15,
16 épocas, batch 8, semilla 42, objetivo 0.5 teacher + 0.5 consenso). SWA all-data sobre las épocas que eligió la SWA de
fold 0 de exp078: **[1, 4, 6]**. Kernel = `exp074-student224` con solo checkpoint y cabeza cambiados (ventanas: el kernel
padre ya usa K_EVAL = 64, las mismas que aquí; backbone, preprocesado, fp16 y blend iguales).

**Criterio (fijo antes de enviar):** `worked` si LB ≥ **0.950**; si no `didnt`. Gold all-data es solo informativo
(gold no entra al entrenamiento ni elige nada). Se reporta el tiempo de la validación contra el del padre.

**Presupuesto:** ≈ 0.2 GPU-h (embeddings all-data + cabeza), compartido con exp080; 1 envío.

**Riesgos:** fold 0 ya dijo que la Bi-GRU no supera a la bolsa MIL (y gold peor en 3 semillas); lo esperable es LB ≈ 0.948.
GRU en fp16 en T4 (se valida en Kaggle).

## 2. PLAN
- Notebook `sub078_1_bigru_alldata.ipynb` = notebook de exp078 sin cambios de código; corre con `ALL_DATA = True`,
  `SWA_EPOCHS = [1, 4, 6]`. Guarda `full_bigru_alldata.pt` (backbone + cabeza) → dataset privado → kernel privado
  `sub078-1-bigru224` → validación COMPLETE → `orq/seqhead/CANDIDATE_bigru.json` y cola del 18-oct.

## 3. VERDICT (números de `results/sub078_1_bigru_alldata/alldata/metrics.json`)

**Resultado:** pendiente de LB · kernel privado `sub078-1-bigru224` v1 validado (COMPLETE, 3 × 13, sin NaN, sin
Traceback/SystemExit/FALLBACK)

| Métrica | Padre LB (exp074-student224) | Este |
|---|---|---|
| gold all-data (SWA ép [1, 4, 6], informativo) | — | 0.9200 |
| GPU-h (embeddings all-data + cabeza) | — | 0.122 |
| tiempo de validación en Kaggle (3 estudios) | DONE 12 s | DONE 13 s |
| LB (criterio worked ≥ 0.950) | 0.948 | pendiente |

Cabeza Bi-GRU 1.35 ms/estudio en fp16 (MIL 0.23 ms): ≈ +1.1 ms por estudio; mismo backbone y mismas 64 ventanas.

**Qué aprendimos:** (pendiente del LB)
