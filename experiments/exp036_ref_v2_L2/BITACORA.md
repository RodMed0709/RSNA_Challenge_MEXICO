# Bitácora — exp036_ref_v2_L2

Estado al 2026-10-11: **SPEC escrita, smoke y screen fold 0 en el pod rsna-e**. Ningún número de resultado existe hasta que
haya `metrics.json`.

## 1. Qué es y para qué sirve

Referencia combinada: exp012 (entrada v2) + etiquetas de entrenamiento L2 (exp021) + 12 épocas (exp015/exp020).
Decide la pseudo contra L2 del fold 0 frente a exp021 (0.817 + 0.01) con veto gold (≥ 0.792). El LB se mide después.

## 2. Datos

- Cache v2_05mm y `study_meta_v2.csv` como exp012; L2 como exp021 (`llm_labels_v4_blend.csv`, cubre los 4,348 estudios del
  pool v1).

## 3. Registro

### 2026-10-11 · Construcción
Notebook copiado de exp012. Cambian celdas 0 (`EXP`, `NOTES`, `EPOCHS = 12`, `TRAIN_LABELS_CSV`, `PARENT_RESULTS_DIR`),
2 (`pool_train` con L2 para el loader de entrenamiento; `pseudo_l2_bin`) y 5 (pseudo contra L2 del modelo y del padre).
Celdas 3 y 4 intactas.
