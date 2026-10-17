# exp079_compact_destilacion_0957

| Campo | Valor |
|---|---|
| Autor | rodrigo |
| Padre | exp066_compact_f0 (compact sola fold 0: pseudo L2 0.8755, gold 0.8972) |
| Fecha | 2026-10-17 |
| Rama | exp079_compact_destilacion_0957 (desde main a139cc8) |
| Estado | spec |

Idea: la destilación que funcionó en exp074 (Raptor 224) aplicada a la compact ConvNeXt-S (receta exp066/exp067):
objetivo = 0.5 × maestro 0.957 en probabilidad + 0.5 × consenso exp031.

Antecedentes: **exp074_raptor224_distill** (mismo objetivo sobre Raptor 224: pseudo f0 SWA 0.8889 → 0.8987, gold
0.9096 → 0.9197; all-data LB 0.948). **exp070_compact_destilacion** (SPEC sin correr, 14-oct) proponía destilar en la compact
con otro maestro (R7 fuera de fold por hash + OOF 5-fold de exp045, media de rangos); no se corrió y su maestro difiere,
así que no se reescribe: este es un experimento nuevo con el maestro de exp074.

Autorizado por Rodrigo el 17-oct ("haz un agente dedicado... un pod dedicado... y déjale un submission dedicado").

## 1. SPEC (antes de tocar el notebook)

**Hipótesis:** la compact (receta exp066) entrenada con objetivos 0.5 × maestro 0.957 + 0.5 × consenso sube la pseudo
fold 0 ≥ +0.005 sobre exp066, como pasó en exp074 con el Raptor 224 (+0.0098): el maestro ve la imagen y da objetivos más
finos y más alineados con la definición del test que el informe solo.

**Única variable cambiada:** los objetivos de entrenamiento.

| Parámetro o celda | Padre (exp066) | Este experimento |
|---|---|---|
| Objetivo (celda 3) | consenso exp031 (Raptor+L2+L3+L4), luego `L|prior` y `L|lr_prior = odds(p)/odds(base)` | p = 0.5 × maestro + 0.5 × consenso exp031 (fila del maestro con NaN → consenso solo), luego el mismo `L|prior` / `L|lr_prior` |

Maestro = `wt_exp074_raptor224_distill/results/exp074_raptor224_distill/teacher.csv` (ya calculado por exp074, no se
recalcula): media en probabilidad 0.5/0.5 de R7 compact (heliosli `rsna-knee-compact-r7cnxfsss`, 4 folds promediados como
en su kernel) y Raptor nartaa 384 (`..._d96_r384_swa.pt`, TTA del slot Raptor de `r7-raptor-repro`), sobre los 4407
estudios; 11 estudios sin R7 (NaN). Gold-58 del maestro (teacher_report.json de exp074): 0.9353. Las probabilidades no van
a git. `TEACHER_W = 0.5` (= exp074).

Todo lo demás igual a exp066: split `split_v1_seed42` (train folds 1–4, 3477; val fold 0), trainer heliosli
`compact_train.py` (sha 14767c7a…, port de folds de exp066 sin cambios), brazo `E_lrprior`, ConvNeXt-S in22k, 352 px,
16 épocas, batch 2, 176 ventanas, LR 1e-4 / 1e-3, WD 0.05, 3 cortes, phys 3.5 mm, atención espacial, semilla 0, último
modelo (sin selección), predicción fold 0 + gold con las funciones del trainer.

Cambios de ejecución declarados (no de modelo):
- `COMPILE = True` (exp066 corrió sin compile; exp067/exp068 con compile). Solo velocidad (~30 %); si el smoke con compile
  falla, se corre sin compile (como `worker_compact2.sh`).
- Caché px384_v4 única de los 4407 estudios en el disco del pod (mismo `build_px_cache.py` de exp066); el pool de
  entrenamiento lo fijan los targets (folds 1–4), igual que con la caché de train de exp066.

**Efecto esperado:** Δ pseudo fold 0 ≈ +0.005 a +0.010 (exp074 dio +0.0098); mismo costo de entrenamiento e inferencia
que exp066.

**Criterios (fijos, no cambian después de correr):** métrica = la de exp066 (`metrics.json` → `pseudo_auc` = compact sola,
pseudo L2 fold 0; `gold_auc` = compact sola en gold-58).
- `worked` si pseudo fold 0 ≥ 0.8805 (exp066 0.8755 + 0.005).
- `killed` si gold < 0.8772 (exp066 0.8972 − 0.02), diga lo que diga pseudo.
- Si no: `didnt`. Gold nunca elige época, checkpoint ni pesos. No hay 5 folds antes del cierre.
- El blend con exp045 (criterio de exp066) se sigue calculando solo como información.

**Fuga declarada:** el maestro vio el fold 0 en su propio entrenamiento (R7 entrenó con ~3/4 de casi todo el pool no-gold
por hash; nartaa con todo). Sus probabilidades en fold 0 son in-sample, pero los objetivos de fold 0 no entran al alumno
(train = folds 1–4); aun así el alumno aprende de un maestro que memorizó parte de la distribución de fold 0 vía estudios
parecidos, así que la pseudo fold 0 puede salir algo inflada. Gold (fuera del pool de entrenamiento de R7 y nuestro) veta
y el LB decide.

**Pregunta de LB (fijada antes de entrenar):** kernel privado `medellinrodrigo/r7-raptor-exp079` = kernel
`medellinrodrigo/r7-raptor-exp067` (final A, LB 0.957) cambiando **solo** la pieza compact exp067 por el all-data destilado
de exp079 (mismo worker, mismo preprocesado, mismos pesos: slot compact = rank-mean por hallazgo R7 0.75 / exp079 0.25;
resto del 0.957 sin cambios). `worked` en LB si LB público ≥ 0.958; si no, la final A sigue siendo `r7-raptor-exp067`.
Solo se construye si fold 0 = `worked` y el all-data existe; el orquestador lo encola (CANDIDATE.json).

**All-data (solo si fold 0 = `worked`):** mismos objetivos y receta que exp067 (folds 0–4, 4348 estudios, fold ficticio
único), en el mismo pod tras el fold 0. **Épocas reducidas por presupuesto** (declarado antes de entrenar): el presupuesto
RunPod de este experimento es ≤ $14 y fold 0 (16 ép) + all-data (16 ép) en RTX PRO 4500 ≈ 26 h ≈ $20. Regla fija, solo de
recursos (no mira resultados): `EPOCHS_ALL = min(16, floor((H_MAX − h_pod_transcurridas − 0.5) / t_ep_all))`, con
`t_ep_all` = segundos por época medidos en el fold 0 × 4348/3477 y `H_MAX` = horas de pod que caben en $13.6 al precio del
pod. Si `EPOCHS_ALL < 6`, no se entrena all-data y no hay pregunta de LB. El scheduler coseno se ajusta solo al número de
épocas (es el del trainer). La pieza de LB difiere entonces de exp067 en objetivos **y** épocas: si sale < 0.958 no se
puede separar una cosa de la otra (se dirá así en el verdict); si sale ≥ 0.958, la pieza sirve tal cual.

**Presupuesto:** 1 pod, 1 GPU RTX PRO 4500 ($0.72/h + disco 420 GB). Caché ≈ 0.5 h · smokes ≈ 0.4 h · fold 0 ≈ 9.6 h
(compile; exp066 sin compile 13.7 h) · all-data ≈ 6 h (8 ép) · total ≤ $14 (watchdog de tiempo para el pod a `H_MAX`).
Congelación 19-oct 12:00 UTC: fin estimado ≈ 18-oct 10:00 UTC.

**Riesgos:**
- Fuga del maestro (arriba).
- Calibración: R7 da probabilidades casi binarias y nartaa 384 moderadas; la media en probabilidad hace que R7 domine la
  forma del objetivo (igual que exp074). `lr_prior` usa odds(p)/odds(base) con p recortado a [0.01, 0.99] (= exp066).
- Licencias: destila pesos de heliosli (R7, "other") y nartaa ("other", OAI). Revisar antes de elegirlo como final de
  premio. Código y pesos de heliosli nunca a git.
- La compact es parte del propio maestro (R7 es una compact): el alumno puede acabar muy correlado con R7 y aportar menos
  diversidad en el slot compact del 0.957 que exp067.

## 2. PLAN

- Celda 0 (markdown) y celda 1 (parámetros): `EXP`, `PADRE`, `TEACHER_CSV`, `TEACHER_W = 0.5`, `ALL_DATA = False`,
  referencias de exp066 (`REF_PSEUDO_COMPACT`, `REF_GOLD_COMPACT`), `EXPORT_DIR`, rutas de la caché única, `COMPILE = True`.
- Celda 3 (datos y targets): mezcla 0.5 maestro + 0.5 consenso antes de construir `L|prior` / `L|lr_prior`
  (mismo bloque que exp074). Con `ALL_DATA = True`: folds 0–4 con fold ficticio 1 (= exp067).
- Celdas 5–7: nombres de salida por modo; con `ALL_DATA = False` la decisión automática usa los criterios de esta SPEC
  (compact sola vs exp066); con `ALL_DATA = True` la evaluación/export es la de exp067 (gold veto + export Kaggle), salida
  en `results/exp079_compact_destilacion_0957/alldata/`.
- Smoke (`SMOKE = True`, 1 época, 48 estudios) en los dos modos antes del fold 0. Cadena autónoma en el pod: caché → smokes
  → fold 0 (train + eval) → commit local de `metrics.json` → si `worked` y caben épocas: all-data + export → podStop.

## 3. VERDICT (números de `results/exp079_compact_destilacion_0957/metrics.json`)

(pendiente)
