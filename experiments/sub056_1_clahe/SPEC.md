# sub056_1_clahe

| Campo | Valor |
|---|---|
| Autor | rodrigo |
| Padre | exp056_raptor_combo |
| Fecha | 2026-10-12 |
| Rama | sub056_1_clahe (desde exp056_raptor_combo, 2083da0) |
| Estado | SPEC · en cola (rsna-e2 ~11:10 UTC) |

Idea: paso 7, sub-experimento de adición al combo. exp049 (CLAHE por corte) quedó al borde contra exp031
(pseudo L2 0.8864, +0.0028 vs umbral +0.003; gold 0.9083). Se prueba si suma encima del combo.

## 1. SPEC (antes de tocar el notebook)

**Hipótesis:** el corpus con CLAHE por corte (clip 2.0, tiles 8×8; el de exp049) sube la pseudo L2 fold 0 del combo
exp056 en ≥ +0.003.

**Única variable:** `CORPUS = /root/corpus_clahe` (CLAHE aplicado offline al uint8 384 de corpus96_384; afecta a train,
validación y gold, y obligaría a aplicarlo también en la inferencia de Kaggle). Todo lo demás = exp056.

**Criterios (fijos, no cambian después de correr):**
- `worked`: pseudo L2 fold 0 (mejor época) ≥ exp056 fold 0 + 0.003 y gold ≥ gold exp056 fold 0 − 0.02.
- `killed`: gold < gold exp056 − 0.02, o pseudo ≤ exp056 − 0.005.
- `didnt`: cualquier otro caso.
(Los valores de exp056 fold 0 salen de `results/exp056_raptor_combo/metrics.json` cuando termine, ~11:40 UTC.)

**Solo fold 0.** Si funciona, se decide aparte (con el orquestador) si entra en el combo; no cambia los folds 1–4 en curso.

**Presupuesto:** ≈ 2.5 GPU-h en rsna-e2 (único pod con /root/corpus_clahe).

## 2. PLAN

- Notebook `sub056_1_clahe.ipynb` = copia del de exp056; celda 0: `EXP`, `PADRE`, `CORPUS`, `NOTES`.
- Smoke (1 época) y fold 0 encadenados en rsna-e2
  cuando termine sub045_1 (comprobar pgrep/nvidia-smi).

## 3. VERDICT

Pendiente.
