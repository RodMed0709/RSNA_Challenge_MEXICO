# SPEC — stress_test v1 (alcance modesto, descriptivo)

Fecha: 2026-10-21 · Fase SDD: SPEC confirmada por Rodrigo (opción 1) → PLAN/IMPLEMENT/RUN con inferencia únicamente.
Reemplaza el alcance de `SPEC.md` (borrador). Lo que `SPEC.md` pedía y esta versión NO hace: OOD real, transferencia
inferencial al LB, lockbox de varias familias. Motivo: auditoría de cobertura del 21-oct (ver `RESULTS.md`).

## 1. Pregunta y alcance

Pregunta: ¿los cambios de receta que mejoran la robustez a perturbaciones de imagen también mejoran (o cuestan) AUC limpia
y se ven en el LB? Solo descriptivo; no se afirma nada sobre private LB ni hospitales externos.

- Solo inferencia. Cero entrenamiento (freeze vigente hasta 22-oct 23:59 UTC).
- Solo checkpoints de familia Raptor (CoAtNet d96/384, corpus `raptor96_384`, inferencia idéntica a exp040) cuyo
  estudio evaluado estuvo FUERA de su entrenamiento: `*_f{k}.pt` evaluados en el fold k de `split_v1_seed42`.
- Panel R (receta): todos los Raptor fold 0 con `best_f0.pt`/`swa_f0.pt` sin fuga documentada (lista final la fija la
  auditoría de Codex y se registra antes de correr; se excluyen exp074/079/080/090 y derivados por fuga del maestro,
  los all-data y los tanques).
- Panel F (5 folds): exp045 y exp054, folds 0-4 (4348 estudios), para casos difíciles y estabilidad.
- Gold-58: no entra en ningún panel de descubrimiento; se reporta aparte, sin influir en nada.
- Etiqueta principal: pseudo L2 binarizada > 0.5 (comparable con exp040). Sensibilidad: v1. Se registra la fuente.

### Panel S (añadido 21-oct por pedido de Rodrigo): modelos de los envíos recientes, estabilidad sin etiquetas

Los ~20 envíos más recientes (tanques, all-data, alumnos) no tienen OOF, así que NO se les calcula AUC de evaluación ni se usan
para dificultad. Sí se mide **estabilidad de predicciones** (no usa etiquetas): para cada condición, `MAD` = media sobre estudios y
hallazgos de |p_cond − p_ref|, y `rho` = Spearman medio entre hallazgos de p_cond y p_ref. Estudios: val fold 0 (871; son in-sample
para all-data y tanques → se marca `in_sample`, solo descriptivo) y gold-58 por separado (fuera del entrenamiento salvo en tanques).
Alcance: checkpoints Raptor-compatibles (misma entrada que exp040: 384 o su variante 224 con el pipeline propio) de los envíos desde el 15-oct
(tank-seeds, tank-v6, tank-v6soft por semilla; exp074/080/090 alumnos; exp056/057/071/075 all-data; r7-raptor si el Raptor propio es
cargable). Pesos externos (heliosli/nartaa) y compact no entran salvo que Codex demuestre que el pipeline exp040 los carga sin cambios;
si no, `not_run` con razón. Seeds de un mismo tanque se reportan por semilla y como ensamble (media de probabilidades).
Criterios fijos: `S_plaus` = media de MAD plausibles, `S_sev` = media de MAD severas (menor = más estable). Se compara por pares
dentro de familia (tank-seeds vs v6 vs v6soft; alumnos vs maestro). Sin correlación con LB (n < 8): tabla con LB al lado.

## 2. Condiciones (subconjunto del catálogo exp040, mismas definiciones y semilla `crc32(uid|cond)`)

| Nivel | Condiciones |
|---|---|
| control | ref (k24); tta_mirror (control, no es estrés) |
| plausible | gauss2, gamma0.7, gamma1.4, contr-20, bias, cortes50, shift+2, k12 |
| severe (diagnóstico, no criterio de selección) | gauss5, rician5, cortes25, res192, drop_s0, drop_sag |

`plausible` y `severe` nunca se promedian juntos. Cada checkpoint: 16 pasadas sobre los estudios de su fold de validación.

## 3. Métricas (definidas antes de ver resultados)

Por checkpoint: AUC macro limpia; Δ por condición vs ref con IC95 bootstrap pareado por estudio (1000 réplicas, semilla 42);
`R_plaus` = media de Δ plausibles; `R_sev` = media de Δ severas; `worst_plaus` = mínimo Δ plausible.
Un hallazgo entra al macro de una celda solo con ≥ 30 positivos y ≥ 30 negativos; si no, se excluye y se informa n.
Mínimo 20 estudios para publicar una celda en `metrics.json` versionable (privacidad).

## 4. Hipótesis y criterios (no cambian tras ver resultados)

- H1: la AUC limpia del panel R no explica la robustez: |Spearman(AUC_limpia, R_plaus)| < 0.5 entre checkpoints. (Descriptiva; n pequeño → se
  reporta rho con IC bootstrap sobre checkpoints y se declara "no concluyente" si el IC incluye 0 y a la vez ±0.5.)
- H2: las recetas con augmentation de intensidad/artefactos/espacial (exp045, 046, 062, 063, 064 si están en el panel) tienen `R_sev` mejor (Δ menos negativo)
  que su padre por > 0.003 y con IC pareado que excluye 0, sin perder más de 0.005 de AUC limpia.
- H3 (hard cases, panel F): sobre el 10 % de estudios más difíciles según OOF de exp045 (pérdida BCE media por estudio, sin gold, etiquetas 0.5/NaN excluidas),
  la diferencia exp054 − exp045 se reporta solo como Δ pareado, nunca AUC absoluta. exp045 y exp054 son la misma familia → panel `provisional`.
- Tolerancia del gate de validez: ref reproduce el pseudo AUC de fold 0 del propio experimento con |Δ| < 0.002.
- LB: se listan las que tienen score (exp029 0.933, exp031 0.934, exp049 0.932, exp045 5f 0.941, exp075 0.941, exp045+054 10f 0.942);
  no se calcula correlación (3-5 familias, < 8). Solo tabla.

## 5. Datos y privacidad

UIDs, predicciones por estudio, etiquetas y manifests: solo en `/workspace/users/rodrigo/results/<expNNN_stress_v1>/` (volumen), nunca en Git/Kaggle.
En Git: SPEC, bitácora, scripts declarados en el PLAN y `metrics.json` agregado (celdas con n ≥ 20). Licencia MIRA: sin redistribuir datos.

## 5b. Ranking y regla de subida a Kaggle (fijada 21-oct, antes de ver resultados)

- Ranking del Panel S: por `S_plaus` (menor = más estable), con `S_sev` como desempate y las dos tablas siempre visibles; ranking aparte para ensambles derivados.
- Un envío NO se justifica solo por ser más estable. Para subir un candidato a Kaggle debe cumplir TODO:
  1. Es un artefacto nuevo (ensamble o combinación de pesos ya entrenados; no un reenvío de algo con LB conocido) y su única variable respecto a la referencia es la ponderación/selección guiada por estabilidad (una variable, SDD).
  2. Tiene reducción relativa de `S_plaus` ≥ 20 % y de `S_sev` ≥ 20 % frente a su referencia de familia (tank-seeds 0.957), y no pierde pseudo AUC limpia / gold en más de 0.005 donde sea medible.
  3. Se registra como experimento numerado nuevo con criterio de LB fijado antes de enviar: `worked` solo si LB ≥ 0.959 y > 0.957 (mismo criterio que tank-v6).
  4. Sin entrenar (freeze), máx. 5 envíos/día UTC, no toca la selección de finales (se eligen a mano en Kaggle).
- Los envíos que ya tienen LB (tank-seeds/v6/v6soft = 0.957) no se resuben.

## 6. Presupuesto y gates

- Tope duro: 14 USD de RunPod (saldo 21-oct: 25.26 USD; ampliado por el Panel S). Hasta 2 GPUs en paralelo `NVIDIA RTX PRO 4500 Blackwell` si acelera; apagar y borrar pods al terminar. Estimado ~10-14 GPU-h. Prioridad si hay que recortar: Panel S de tanques y los 5 envíos más recientes > Panel R > Panel F. Plazo: resultados antes del 22-oct 12:00 UTC.
- Gates: (1) smoke 20 estudios; (2) ref reproduce pseudo AUC; (3) conteos por panel/condición sin IDs; (4) si el gasto proyectado > tope, parar y reportar.
- Verdict en sección 3 de la SPEC del experimento y fila en `docs/EXPERIMENTOS.md`.
