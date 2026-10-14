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

### 2026-10-11 · Smoke (rsna-e, `results/exp036_ref_v2_L2/metrics_smoke.json`)
`SMOKE = True` corrió completo hasta "Guardado" (git 09c7b24; el primer intento falló en `current_user()` por correr desde
un worktree sin `RSNA_USER`: se relanzó con `export RSNA_USER=rodrigo`). Celda 2: pool v2 4,348 estudios, L2 en 4,348,
0 fallback v1. 1 época, 80 train / 40 val: pseudo v1 0.573, contra L2 0.553, gold 0.542; padre (exp012) contra L2 sobre
esos 40: 0.784. Números sin sentido con 1 época; solo prueban el pipeline. 0.010 GPU-h.

### 2026-10-11 · Run real lanzado
Primer intento (PID 2836, 16:42) murió sin traceback al cerrar la época 1 (~16:52): papermill y kernel desaparecieron,
`history_f0.csv` y el `executed_*.ipynb` quedaron en 0 bytes; sin OOM en `memory.events` del contenedor. Dos relanzamientos
inline por ssh murieron al instante (logs de 0 bytes). Logs fallidos en `results/exp036_ref_v2_L2/failed_*`.
Relanzado desde un script en el pod (`setsid nohup bash /root/launch036.sh`, que llama a `run_notebook.sh ... -p SMOKE False
-y 'FOLDS: [0]'` con `RSNA_USER=rodrigo`): PID 5680, log `wt_exp036/results/exp036_ref_v2_L2/run_20261011_165655.log`.
Época 1 cerrada a los 10.0 min (17:07). GPU compartida con exp035.
