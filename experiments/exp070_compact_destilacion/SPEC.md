# exp070_compact_destilacion

| Campo | Valor |
|---|---|
| Autor | rodrigo |
| Padre | exp066_compact_f0 (compact sola fold 0: pseudo L2 0.8755, gold 0.8972) |
| Fecha | 2026-10-14 |
| Rama | exp070_compact_destilacion (desde exp066_compact_f0) |
| Estado | spec (no lanzado: falta generar los targets del maestro) |

Idea: destilar el blend público fuerte (R7 + Raptor, LB 0.957) en un alumno barato: la compact (o una Raptor 224) entrenada
con las predicciones del maestro como etiquetas blandas en lugar de nuestro consenso LLM.

## 1. SPEC

**Maestro (sin fuga en fold 0):**
- R7: los 4 modelos públicos usan folds `sha1(uid)[:8] % 4` (línea `pool_fold` del trainer R7, verificada en la fuente
  pública). Para cada estudio se usa solo el modelo que NO lo vio → predicción fuera de fold de R7 para los ~4344 estudios del
  pool de heliosli (los demás: media de los 4, marcados).
- Raptor: los pesos de nartaa son all-data (sin OOF), así que el lado Raptor del maestro es nuestro **OOF 5-fold de exp045**
  (`oof_swa_fK.csv`, OOF 0.8844), que sí es fuera de fold en nuestro split.
- Maestro = media de rangos por hallazgo 0.5 R7-OOF + 0.5 exp045-OOF, convertida a probabilidad con la prevalencia del
  consenso (`p = cuantil del consenso en el mismo rango`), y luego a `lr_prior` como en exp066.

**Hipótesis:** la compact entrenada con el maestro (folds 1–4) supera a exp066 en pseudo L2 fold 0 ≥ +0.005, porque el
maestro es mejor que el consenso LLM en rango (pseudo exp045 OOF 0.8844 > compact 0.8755).

**Única variable:** targets de entrenamiento = maestro (en vez del consenso exp031). Mismo trainer/receta/caché que exp066.

**Fuga a vigilar:** el lado exp045 del maestro usa fold 0 solo vía `oof_swa_f0` (modelo que no vio fold 0), pero los targets
de fold 0 no entran al alumno (train = folds 1–4). Pseudo L2 sigue siendo la vara; gold solo veta.

**Criterios (fijos):** `worked` pseudo ≥ 0.8805 y gold ≥ 0.8772; `killed` gold < 0.8772; `didnt` otro caso.
Si `worked` → all-data (SPEC aparte) y variante Raptor 224 (alumno eficiente para el track de eficiencia).

**Presupuesto:** maestro R7-OOF ≈ 1.5 GPU-h (4348 estudios, 1 forward por estudio, caché all-data de `rsna-c67`); alumno
≈ 14 GPU-h (como exp066).

## 2. PLAN (pendiente de lanzar)

1. Tras exp067/exp069 en `rsna-c67` (caché all-data todavía en disco): script de inferencia R7 por fold (Net R7 con
   `state_spec` público, `extent`, `syn_decomp`, carga estricta) → `results/exp070_compact_destilacion/teacher_r7_oof.csv`.
2. Celda 2 del notebook (copia de exp066): targets = maestro; resto igual. Pod 400 GB como exp068.

## 3. VERDICT

(pendiente)
