# sub056_1_clahe

| Campo | Valor |
|---|---|
| Autor | rodrigo |
| Padre | exp056_raptor_combo |
| Fecha | 2026-10-12 |
| Rama | sub056_1_clahe (desde exp056_raptor_combo, 2083da0) |
| Estado | didnt (fold 0) |

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

## 3. VERDICT (números de `results/sub056_1_clahe/metrics.json`)

**Resultado:** `didnt` · **Fase alcanzada:** screen fold 0 (rsna-e2, 11:09–13:40 UTC)

| Métrica (fold 0) | exp056 (padre) | sub056_1 | Δ |
|---|---|---|---|
| pseudo L2 (mejor época) | 0.8857 (ep 15) | 0.8839 (ep 15) | -0.0018 |
| pseudo L2 (SWA top-3) | 0.8854 | 0.8836 | -0.0018 |
| v1 | 0.8407 | 0.8425 | +0.0018 |
| gold (veto) | 0.9198 | 0.9143 | -0.0055 |
| GPU-h | 2.51 | 2.48 | — |

Criterio: worked exige Δ pseudo ≥ +0.003 → no; killed exige Δ ≤ −0.005 o gold < 0.8998 → no. → `didnt`.

**Qué aprendimos:** el CLAHE por corte, que contra exp031 quedó al borde (+0.0028), no suma sobre el combo (−0.0018,
dentro de ~1.4× el ruido de semilla) y baja gold −0.005. No se apila; no hay candidato para el 15-oct por esta vía.
Además obligaría a aplicar CLAHE en la inferencia de Kaggle (coste de CPU).

**Siguiente paso sugerido:** ninguno con CLAHE. Pod rsna-e2 detenido al terminar.
