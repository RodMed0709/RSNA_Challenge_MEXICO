# Bitácora — exp035_etiquetas_combinadas

Estado al 2026-10-11: **SPEC escrita, smoke y screen fold 0 en el pod rsna-e**. Ningún número de resultado existe hasta que
haya `metrics.json` y envío a Kaggle.

## 1. Qué es y para qué sirve

exp021 con una sola variable cambiada: etiquetas de entrenamiento L2 → promedio por celda de L2, L3 y L4 (las tres
blandas en [0, 1]). Decide el LB público del fold 0 contra exp021 (0.836).

## 2. Datos (comprobado en rsna-e, 2026-10-11)

- L2 `stevenleehans/.../llm_labels_v4_blend.csv`: 4,407 estudios, [0.005, 1], 211 valores; pool 4,348/4,348, gold 58/58.
- L3 `pilkwang/.../report_labels_v2.csv`: 4,406 estudios, 5 niveles {0.08, 0.28, 0.68, 0.82, 0.94}; pool 4,348/4,348, gold 57/58.
- L4 `lixin73/.../labels_llm_gpt56sol.csv`: 4,407 estudios, [0, 1], 32 valores; pool 4,348/4,348, gold 58/58.
- Celdas del pool con 3 fuentes: 52,176/52,176 (0 con menos, 0 fallback v1). Media del objetivo: L2 0.336, L3 0.349, L4 0.259.
- Acuerdo binarizado (> 0.5) en el pool: L2–L4 99.2 %, L2–L3 95.4 %, L3–L4 95.0 % (L2 v4_blend ya incluye GPT-5.6-Sol = L4).

## 3. Registro

### 2026-10-11 · Construcción
Notebook copiado de exp021. Cambian celdas 0 (`EXP`, `PADRE`, `NOTES`, `L2_CSV`/`L3_CSV`/`L4_CSV`, `PARENT_RESULTS_DIR`),
2 (`pool_train` con el promedio; `pseudo_l2_bin`, `pseudo_avg_bin`) y 5 (pseudo contra L2 y contra el promedio, del
modelo y del padre). Celda 4 intacta.

### 2026-10-11 · Smoke (rsna-e, `results/exp035_etiquetas_combinadas/metrics_smoke.json`)
`SMOKE = True` corrió completo hasta "Guardado" (git 857af32; el primer intento falló en `current_user()` por correr desde
un worktree sin `RSNA_USER`: se relanzó con `export RSNA_USER=rodrigo`). Celda 2: L2/L3/L4 4,348/4,348, celdas con 3 fuentes
52,176, 0 fallback v1. 1 época, 80 train / 40 val: pseudo v1 0.496, contra L2 0.515, contra promedio 0.517, gold 0.530;
padre (exp021) sobre esos 40: contra L2 0.820, contra promedio 0.815. Números sin sentido con 1 época; solo prueban el
pipeline. 0.005 GPU-h.

### 2026-10-11 · Run real lanzado
`setsid nohup bash run_notebook.sh ... -p SMOKE False -y 'FOLDS: [0]'` con `RSNA_USER=rodrigo`, PID 2805,
log `wt_exp035/results/exp035_etiquetas_combinadas/run_20261011_164212.log`. GPU compartida con exp036.
