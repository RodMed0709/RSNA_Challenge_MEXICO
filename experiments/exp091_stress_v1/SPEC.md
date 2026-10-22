# exp091_stress_v1

| Campo | Valor |
|---|---|
| Autor | rodrigo |
| Padre | exp040_robustez |
| Fecha | 2026-10-21 |
| Rama | exp091_stress_v1 |
| Estado | worked |

Idea: benchmark descriptivo de robustez, solo inferencia, sobre checkpoints existentes. Esta SPEC aplica
`stress_test/SPEC_v1_modesto.md` y su addendum de Panel S; no entrena ni selecciona checkpoints con gold.

## 1. SPEC (confirmada antes de implementar)

**Pregunta:** ¿los cambios de receta de la familia Raptor cambian la estabilidad ante perturbaciones plausibles/severas,
y cómo se relaciona descriptivamente ese cambio con AUC limpia y LB ya observado?

**Única variable frente a exp040:** el checkpoint evaluado pasa de un único `exp029 best_f0.pt` a tres paneles
predeclarados (R/F/S), manteniendo el catálogo reducido de condiciones, el split, las semillas de perturbación
`crc32(uid|cond)`, las etiquetas y la definición de métricas de `stress_test/SPEC_v1_modesto.md`.

No se toca `src/knee/`, `split_v1_seed42`, pesos, OOF de origen ni notebooks de otros experimentos. Gold-58 no descubre
casos ni elige checkpoints/condiciones; se reporta aparte. Panel S es estabilidad sin etiquetas y se marca `in_sample`.

### Condiciones fijas (16 resultados por checkpoint)

| Nivel | Condiciones |
|---|---|
| control | `ref` (k24), `tta_mirror` = media(`ref`, pasada `mirror`) |
| plausible | `gauss2`, `gamma0.7`, `gamma1.4`, `contr-20`, `bias`, `cortes50`, `shift+2`, `k12` |
| severe | `gauss5`, `rician5`, `cortes25`, `res192`, `drop_s0`, `drop_sag` |

`plausible` y `severe` nunca se promedian juntos. `mirror` se ejecuta para derivar `tta_mirror`, pero no se publica como
condición adicional. Para checkpoints de 224 px se conserva su pipeline propio (crop central 320 y resize a 224,
slots 0-2); no se les atribuye AUC OOF y solo aparecen en Panel S.

### Panel R congelado (fold 0, OOF sin fuga, mismo pipeline limpio que exp040)

El checkpoint se eligió exclusivamente por la pseudo AUC L2 propia entre `best_f0` y `swa_f0`. Las rutas OOF se usan
solo como prueba de reproducción y etiquetas/predicciones privadas; no se versionan.

| ID | checkpoint | OOF correspondiente | pseudo AUC propia |
|---|---|---|---:|
| exp029 | `/workspace/users/rodrigo/wt_exp029/results/exp029_raptor_entreno/best_f0.pt` | `/workspace/users/rodrigo/wt_exp029/results/exp029_raptor_entreno/oof_best_f0.csv` | 0.868177655 |
| exp030 | `/workspace/users/rodrigo/wt_exp030/results/exp030_raptor_etiquetas_L2/best_f0.pt` | `/workspace/users/rodrigo/wt_exp030/results/exp030_raptor_etiquetas_L2/oof_best_f0.csv` | 0.879464247 |
| exp031 | `/workspace/users/rodrigo/wt_exp031_raptor_etiquetas_consenso/results/exp031_raptor_etiquetas_consenso/best_f0.pt` | `/workspace/users/rodrigo/wt_exp031_raptor_etiquetas_consenso/results/exp031_raptor_etiquetas_consenso/oof_best_f0.csv` | 0.883632344 |
| exp033 | `/workspace/users/rodrigo/wt_exp033_raptor_24epocas/results/exp033_raptor_24epocas/swa_f0.pt` | `/workspace/users/rodrigo/wt_exp033_raptor_24epocas/results/exp033_raptor_24epocas/oof_swa_f0.csv` | 0.871547643 |
| exp045 | `/workspace/users/rodrigo/wt_exp045_raptor_aug_espacial/results/exp045_raptor_aug_espacial/best_f0.pt` | `/workspace/users/rodrigo/wt_exp045_raptor_aug_espacial/results/exp045_raptor_aug_espacial/oof_best_f0.csv` | 0.889889948 |
| exp046 | `/workspace/users/rodrigo/wt_exp046_raptor_aug_intensidad/results/exp046_raptor_aug_intensidad/swa_f0.pt` | `/workspace/users/rodrigo/wt_exp046_raptor_aug_intensidad/results/exp046_raptor_aug_intensidad/oof_swa_f0.csv` | 0.885761400 |
| exp047 | `/workspace/users/rodrigo/wt_exp047_raptor_slot_dropout/results/exp047_raptor_slot_dropout/swa_f0.pt` | `/workspace/users/rodrigo/wt_exp047_raptor_slot_dropout/results/exp047_raptor_slot_dropout/oof_swa_f0.csv` | 0.884930336 |
| exp048 | `/workspace/users/rodrigo/wt_exp048_raptor_mixup/results/exp048_raptor_mixup/swa_f0.pt` | `/workspace/users/rodrigo/wt_exp048_raptor_mixup/results/exp048_raptor_mixup/oof_swa_f0.csv` | 0.884695436 |
| exp052 | `/workspace/users/rodrigo/wt_exp052_raptor_etiqueta_estocastica/results/exp052_raptor_etiqueta_estocastica/best_f0.pt` | `/workspace/users/rodrigo/wt_exp052_raptor_etiqueta_estocastica/results/exp052_raptor_etiqueta_estocastica/oof_best_f0.csv` | 0.885057021 |
| exp053 | `/workspace/users/rodrigo/wt_exp053_raptor_etiqueta_temperatura/results/exp053_raptor_etiqueta_temperatura/swa_f0.pt` | `/workspace/users/rodrigo/wt_exp053_raptor_etiqueta_temperatura/results/exp053_raptor_etiqueta_temperatura/oof_swa_f0.csv` | 0.886243779 |
| exp054 | `/workspace/users/rodrigo/wt_exp054_etiquetas_L5/results/exp054_etiquetas_L5/best_f0.pt` | `/workspace/users/rodrigo/wt_exp054_etiquetas_L5/results/exp054_etiquetas_L5/oof_best_f0.csv` | 0.889598301 |
| exp056 | `/workspace/users/rodrigo/wt_exp056_raptor_combo/results/exp056_raptor_combo/best_f0.pt` | `/workspace/users/rodrigo/wt_exp056_raptor_combo/results/exp056_raptor_combo/oof_best_f0.csv` | 0.885671058 |
| exp062_s42 | `/workspace/users/rodrigo/wt_exp062_aug_artefactos_light/results/exp062_aug_artefactos_light/best_f0.pt` | `/workspace/users/rodrigo/wt_exp062_aug_artefactos_light/results/exp062_aug_artefactos_light/oof_best_f0.csv` | 0.888269962 |
| exp062_s7 | `/workspace/users/rodrigo/wt_exp062_aug_artefactos_light_seed7/results/exp062_aug_artefactos_light/best_f0.pt` | `/workspace/users/rodrigo/wt_exp062_aug_artefactos_light_seed7/results/exp062_aug_artefactos_light/oof_best_f0.csv` | 0.888009677 |
| exp063_s42 | `/workspace/users/rodrigo/wt_exp063_aug_artefactos_strong/results/exp063_aug_artefactos_strong/best_f0.pt` | `/workspace/users/rodrigo/wt_exp063_aug_artefactos_strong/results/exp063_aug_artefactos_strong/oof_best_f0.csv` | 0.886003033 |
| exp063_s7 | `/workspace/users/rodrigo/wt_exp063_aug_artefactos_strong_seed7/results/exp063_aug_artefactos_strong/best_f0.pt` | `/workspace/users/rodrigo/wt_exp063_aug_artefactos_strong_seed7/results/exp063_aug_artefactos_strong/oof_best_f0.csv` | 0.887277802 |
| exp072 | `/workspace/users/rodrigo/wt_exp072_raptor_fm_aux/results/exp072_raptor_fm_aux/swa_f0.pt` | `/workspace/users/rodrigo/wt_exp072_raptor_fm_aux/results/exp072_raptor_fm_aux/oof_swa_f0.csv` | 0.892370878 |
| exp075 | `/workspace/users/rodrigo/wt_exp075_raptor_ssl_dino/results/exp075_raptor_ssl_dino/best_f0.pt` | `/workspace/users/rodrigo/wt_exp075_raptor_ssl_dino/results/exp075_raptor_ssl_dino/oof_best_f0.csv` | 0.890443483 |

**Exclusiones auditadas:** exp034 (ConvNeXt, no CoAtNet Raptor); exp032 (alineación L/R cambia la inferencia limpia);
exp043 (entrenado con 3 slots); exp049/051/058/059/061 (preprocesados/corpus distintos); exp050 (entrada completa
384, sin crop 320); exp057 (224 y 3 slots); exp044/064/071 y tanques (all-data sin OOF); exp074/079/080/090 y
derivados (fuga del maestro o arquitectura/pipeline de alumno); pesos externos heliosli/nartaa y compact.

### Panel F congelado (hard cases; exp045 y exp054, folds 0-4)

| ID | checkpoint | OOF | pseudo AUC propia |
|---|---|---|---:|
| exp045_f0 | `/workspace/users/rodrigo/wt_exp045_raptor_aug_espacial/results/exp045_raptor_aug_espacial/best_f0.pt` | `.../oof_best_f0.csv` | 0.889889948 |
| exp045_f1 | `/workspace/users/rodrigo/wt_exp045_f1/results/exp045_raptor_aug_espacial/swa_f1.pt` | `.../oof_swa_f1.csv` | 0.884419225 |
| exp045_f2 | `/workspace/users/rodrigo/wt_exp045_f2/results/exp045_raptor_aug_espacial/swa_f2.pt` | `.../oof_swa_f2.csv` | 0.885823724 |
| exp045_f3 | `/workspace/users/rodrigo/wt_exp045_f3/results/exp045_raptor_aug_espacial/swa_f3.pt` | `.../oof_swa_f3.csv` | 0.880969402 |
| exp045_f4 | `/workspace/users/rodrigo/wt_exp045_f4/results/exp045_raptor_aug_espacial/best_f4.pt` | `.../oof_best_f4.csv` | 0.884910978 |
| exp054_f0 | `/workspace/users/rodrigo/wt_exp054_etiquetas_L5/results/exp054_etiquetas_L5/best_f0.pt` | `.../oof_best_f0.csv` | 0.889598301 |
| exp054_f1 | `/workspace/users/rodrigo/wt_exp054_f1/results/exp054_etiquetas_L5/swa_f1.pt` | `.../oof_swa_f1.csv` | 0.884317437 |
| exp054_f2 | `/workspace/users/rodrigo/wt_exp054_f2/results/exp054_etiquetas_L5/swa_f2.pt` | `.../oof_swa_f2.csv` | 0.882719086 |
| exp054_f3 | `/workspace/users/rodrigo/wt_exp054_f3/results/exp054_etiquetas_L5/swa_f3.pt` | `.../oof_swa_f3.csv` | 0.885592642 |
| exp054_f4 | `/workspace/users/rodrigo/wt_exp054_f4/results/exp054_etiquetas_L5/best_f4.pt` | `.../oof_best_f4.csv` | 0.883586111 |

Las rutas OOF abreviadas pertenecen al mismo directorio que el checkpoint de su fila. El 10 % difícil se define una sola
vez por BCE OOF de exp045 concatenado, excluyendo celdas L2 0.5/NaN. H3 reporta solo Δ pareado exp054−exp045.

### Panel S congelado (17 checkpoints + 3 ensambles derivados)

| ID | checkpoint | pipeline | marca |
|---|---|---|---|
| exp056 | `/workspace/users/rodrigo/wt_exp056_raptor_combo/results/exp056_raptor_combo/swa_alldata.pt` | 320, 5 slots | `in_sample` val f0 |
| exp057 | `/workspace/users/rodrigo/wt_exp057_raptor_eficiente/results/exp057_raptor_eficiente/swa_alldata.pt` | 224, slots 0-2 | `in_sample` val f0 |
| exp071 | `/workspace/users/rodrigo/kaggle_export/exp071/exp071_swa_alldata.pt` | 320, 5 slots | `in_sample` val f0 |
| sub075_1 | `/workspace/users/rodrigo/kaggle_export/sub075_1_alldata/sub075_1_swa_alldata.pt` | 320, 5 slots | `in_sample` val f0 |
| exp074 | `/workspace/users/rodrigo/kaggle_export/exp074_raptor224_distill/swa_alldata.pt` | 224, slots 0-2 | `in_sample` val f0 |
| exp080 | `/workspace/users/rodrigo/kaggle_export/exp080_seqhead_ctrl_mil/full_mil_alldata.pt` | 224, slots 0-2 | `in_sample` val f0 |
| exp090 | `/workspace/users/rodrigo/kaggle_export/exp090_raptor224_v6soft_distill/swa_alldata.pt` | 224, slots 0-2 | `in_sample` val f0 |
| exp085_s42/s7/s123/s2024 | `/workspace/users/rodrigo/kaggle_export/exp085_tank_seeds/swa_alldata_<seed>.pt` | 320, 5 slots | val y gold `in_sample` |
| exp088_s42/s7/s123/s2024 | `/workspace/users/rodrigo/kaggle_export/exp088_tank_v6/swa_alldata_<seed>.pt` | 320, 5 slots | val y gold `in_sample` |
| exp089_s42/s7 | `/workspace/users/rodrigo/kaggle_export/exp089_tank_v6soft/swa_alldata_<seed>.pt` | 320, 5 slots | val y gold `in_sample` |

Además se reporta el ensamble de probabilidades por tanque (`exp085_ens4`, `exp088_ens4`, `exp089_ens2`).
R7+Raptor no entra: sus componentes son nartaa/heliosli/compact externos y no se cargan con el pipeline exp040 sin
código ajeno. Queda `not_run` con esa razón. No se calcula AUC ni correlación de Panel S.

### Métricas e hipótesis fijadas

- R/F: macro AUC limpia y Δ por condición con IC95 bootstrap pareado por estudio (1000, semilla 42); `R_plaus`,
  `R_sev`, `worst_plaus`. Un hallazgo entra solo con ≥30 positivos y ≥30 negativos.
- S: MAD media estudio×hallazgo y Spearman medio por hallazgo contra `ref`; `S_plaus` y `S_sev` son medias de MAD
  dentro de cada nivel (menor es más estable). Val f0 y gold-58 se separan.
- H1: `|Spearman(AUC_limpia, R_plaus)| < 0.5`; no concluyente si su IC incluye 0 y ±0.5.
- H2: exp045/046/062/063 (y exp064 solo si tuviera OOF, que no lo tiene) mejoran `R_sev` frente a su padre por >0.003,
  IC pareado excluye 0 y pérdida limpia ≤0.005.
- H3: en el 10 % difícil de exp045, Δ pareado exp054−exp045; nunca AUC absoluta.
- LB fijo, sin correlación: exp029 0.933; exp031 0.934; exp049 0.932; exp045 5f 0.941; exp075 0.941;
  exp045+054 10f 0.942. Exp049 queda en la tabla aunque no esté en Panel R por pipeline CLAHE distinto.

**Gates (no cambian):** smoke 20 estudios × 3 condiciones × exp029; después `ref` completo de exp029 debe reproducir
0.868177655 con |Δ|<0.002. Cada checkpoint R/F debe reproducir su propia pseudo AUC con |Δ|<0.002 o quedar `invalid`
sin entrar en H1/H2/H3. Celdas versionables requieren n≥20 y nunca contienen UID.

**Presupuesto:** tope duro USD 14, hasta 2 RTX PRO 4500 Blackwell, estimado 10-14 GPU-h. Orden: auditoría → smoke →
Panel S (tanques y cinco envíos más recientes primero) → Panel R → Panel F. Parar si el gasto proyectado supera USD 14.
Plazo solicitado: 2026-10-22 12:00 UTC.

**Riesgos:** base Git disponible en el pod solo hasta `origin/main` a139cc8 (15-oct); pesos repartidos en worktrees;
bf16 no determinista; gold pequeño; Panel S all-data es descriptivo; paneles comparten familia y no prueban OOD/private LB.

## 2. PLAN

Scripts auxiliares declarados (solo inferencia/análisis):

- `panel_config.json`: paneles congelados, paths, fold, resolución/slots, pseudo AUC y exclusiones.
- `infer_stress.py`: reutiliza modelo, ventanas y perturbaciones de exp040; parametriza checkpoint, fold, resolución y slots;
  escribe predicciones privadas reanudables por checkpoint/condición.
- `analyze_stress.py`: gate de reproducción, bootstrap, H1-H3, hard cases, Panel S, tabla LB, figuras y checkpoints
  atómicos de `results/exp091_stress_v1/metrics.json` sin UIDs.
- `chain.sh`: orden autónomo, `status.log`, `HANDOFF.md`, watchdog de coste y limpieza de `guard_skip`/stop del pod.
- `test_stress.py`: tests unitarios para catálogo/semillas, transforms, selección de cohortes, métricas agregadas y privacidad.

Secuencia: tests RED→GREEN → commit SPEC/PLAN → smoke → gate exp029 → Panel S prioritario → Panel R → Panel F →
análisis/verdict. Predicciones/UIDs permanecen en el volumen; a Git solo SPEC, bitácora, scripts, tests, `metrics.json`
y figuras agregadas.

## 3. VERDICT (números de `results/exp091_stress_v1/metrics.json`)

**Resultado:** worked · **Fase alcanzada:** análisis de inferencia S/R/F

- Cobertura: 45/45 checkpoints; completa=true.
- Gate exp029: pseudo AUC 0.868175; Δ -0.000002206; passed=true.
- H1: rho(clean, R_plaus)=-0.172.
- H2: 2/6 comparaciones confirmadas.
- H3: Δ BCE pareado exp054−exp045 en hard cases (ref)=-0.008311.
- Inferencia: 11.198 GPU-h medidas; coste RunPod estimado USD 11.64.

**Qué aprendimos:** ver H1–H3 y tablas agregadas en `metrics.json`; Panel S es descriptivo e in-sample. No se calculó correlación con LB ni se usó gold para seleccionar.

**Siguiente paso sugerido:** conservar estas métricas como diagnóstico; no convertirlas en criterio de selección sin una validación futura fuera de familia/tiempo.
