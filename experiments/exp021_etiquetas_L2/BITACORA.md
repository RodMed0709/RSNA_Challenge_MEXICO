# Bitácora — exp021_etiquetas_L2

Estado al 2026-10-08: **SPEC escrita (pendiente de que Rodrigo confirme la desviación: decide el LB), smoke pasado, screen fold 0
corriendo en el pod A**. Ningún número de este archivo existe hasta que haya `metrics.json` y envío a Kaggle.

## 1. Qué es y para qué sirve

La pseudo AUC interna sube (0.83 → 0.85) y el LB público no se mueve (0.81). La hipótesis es que el problema son las
etiquetas: v1 no sigue las definiciones oficiales estrictas de Kaggle. Este experimento cambia **solo** las etiquetas de
entrenamiento de la receta exp002 (v1 → L2, el dataset público `stevenleehans/rsna-knee-llm-report-labels`, archivo
`llm_labels_v4_blend.csv`) y mira qué pasa en el LB.

**Por qué decide el LB y no la pseudo.** La pseudo mide acuerdo con v1: si las etiquetas de entrenamiento cambian, la
pseudo contra v1 castiga justo lo que queremos probar, y la pseudo contra L2 premia a L2 por construcción. Se reportan las
dos; decide el LB público del fold 0 (ver SPEC). Es una desviación del SDD que Rodrigo debe revisar.

## 2. Datos

- L2 en el volumen: `/workspace/data/labels/public/stevenleehans/rsna-knee-llm-report-labels/llm_labels_v4_blend.csv`
  (lo descargó exp011). 4,407 estudios, 12 columnas, probabilidades en [0, 1] (211 valores distintos; 0.25 muy frecuente).
- Cobertura sobre el pool (folds 0–4): 4,348 / 4,348 con las 12 etiquetas; **0 estudios caen a v1**. Gold: 58 / 58 en L2
  (no se usa para entrenar).
- L2 declara ajustes de Synovitis con los 58 gold: el veto gold queda contaminado (riesgo en la SPEC).

## 3. Registro

### 2026-10-08 · Construcción
Notebook copiado de exp002 (main). Cambian celdas 0 (`EXP`, `PADRE`, `NOTES`, `TRAIN_LABELS_CSV`, `PARENT_RESULTS_DIR`),
2 (`pool_train` con L2 para el loader de entrenamiento; `pseudo_l2_bin` para medir) y 5 (pseudo contra L2 del modelo y
del padre, guardada dentro de `per_fold`). Celda 4 intacta: el checkpoint se elige con pseudo contra v1, como el padre.

### 2026-10-08 · Smoke (pod A, `results/exp021_etiquetas_L2/metrics_smoke.json`)
`SMOKE = True` corrió completo hasta "Guardado" (git a7c13b6). Celda 2: L2 en 4,348 estudios del pool, 0 con fallback v1.
1 época, 80 estudios de train, 40 de val: pseudo contra v1 0.480, contra L2 0.504 (40 estudios), gold 0.535; números sin
sentido con 1 época, solo prueban que el pipeline y los campos nuevos (`pseudo_auc_L2`, `parent_pseudo_auc_L2`,
`N_TRAIN_L2`, `N_TRAIN_FALLBACK_V1`) llegan a `metrics.json`. 0.029 GPU-h; ~20 min de reloj esperando la carga de pesos
del volumen (FUSE saturado por exp018/019/020).

### 2026-10-08 · Run real lanzado
`setsid nohup bash run_notebook.sh ... -p SMOKE False -y 'FOLDS: [0]'`, PID 20119,
log `results/exp021_etiquetas_L2/run_20261008_204325.log`. GPU compartida con exp018, exp019 y exp020.
