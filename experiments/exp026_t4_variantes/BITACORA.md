# Bitácora exp026_t4_variantes

- 2026-10-11 · Worktree `worktrees/RSNA_wt026` desde `exp025_perfil_raptor`. Claim en rsna-c
  (`/workspace/users/.claims/exp026_t4_variantes`; rsna-d ya no existe).
- Estudios (`exp026_studies.csv`, solo en scratchpad, con etiquetas): 58 gold + 92 fold 0 (principales),
  30 fold 0 para el perfilado serie en frío, 2 de calentamiento; semilla 0 sobre `split_v1_seed42`.
  `train.csv` solo trae etiquetas para los 58 gold; L2 = `llm_labels_v4_blend.csv`.
- `make_kernel.py` arma el kernel `bench` (celda del notebook en base64 + `t4_variants.py` + lista de
  estudios) y 4 kernels de envío (`submit_V1..V4`, celda del notebook parcheada; cada parche se exige
  exactamente una vez).
- Smoke en rsna-c (PRO 4500, `/kaggle` simulado con symlinks): 3 y 8 estudios, todo corre; lector
  instrumentado idéntico byte a byte (3/3). Hallazgo: **cudnn.benchmark vuelve a autoajustar en cada hilo
  consumidor nuevo** (caché por hilo): el notebook paga ≈ 13 s (V0) / 22 s (channels_last) / 33 s (V3) por
  GPU y por corrida en la PRO 4500. El arnés reporta wall y "steady" (wall − exceso del primer estudio).
- Kernels de envío V1–V4 probados en rsna-c sobre los 3 estudios de test: escriben `submission.csv` sin
  errores (V1–V3 mismo orden de rangos que el original en 3 estudios; V4 cambia algunos).
- Imagen Docker: sin fijar (los kernels con la imagen fijada del notebook llevaban > 45 min en QUEUED);
  bench y envíos usan la misma imagen (la última de Kaggle).
- Kaggle limita a 2 sesiones GPU batch simultáneas: el push del bench reintenta cada 2 min mientras otro
  agente ocupa las dos (`scratchpad/kaggle_exp026/push_retry.sh`).
- Push del bench a las 11:28 (se liberó un hueco GPU), QUEUED ~25 min, RUNNING 11:55–12:35, COMPLETE,
  2 297 s. Salidas bajadas a `scratchpad/kaggle_exp026/bench_out/` (preds y JSON crudos, no van a git).
- `analyze.py`: el bootstrap con `roc_auc_score` tardaba > 20 min en Windows → AUC Mann-Whitney con
  `rankdata` (igual a sklearn, comprobado) y 1 000 réplicas: 45 s.
- Resultado en SPEC §3. Kernels de envío listos (sin enviar): `scratchpad/kaggle_exp026/submit_V1`
  (AUC) y `submit_V2` (eficiencia); V3 y V4 también están generados pero quedan dominados por V2.
