# exp053_raptor_etiqueta_temperatura — bitácora

- 2026-10-12 · SPEC + notebook (copia de exp031, solo celdas de la variable). Claim en /workspace/users/.claims. Pod rsna-g ocupado por sub031_1 (ep3/16 a las 02:13 UTC); smoke y run en cuanto termine.
- Pod rsna-g: worktree `wt_exp053_raptor_etiqueta_temperatura`, symlink del padre; en la cola `/root/queue_t4a.sh` tras exp052.
- 04:13 UTC smoke rc=0: T=0.7 → media del objetivo 0.302 → 0.282, celdas en (0.2, 0.8) 0.200 → 0.139, binarización > 0.5 idéntica (1.0000). Aviso inofensivo de papermill ("Unable to parse line 32", comentario de `LABEL_T`; no se pasa por -p).
- 06:47 UTC run real fold 0 (log `run_20261012_0647*.log`); 09:20 UTC fin (rc=0): mejor época 15, pseudo L2 0.8854, SWA 0.8862, gold 0.9121, 2.53 GPU-h → `didnt`. Varas extra (best): Raptor 0.8933 · L3 0.8739 · L4 0.8851 · v1 0.8409. |Δ| < 0.003 → `/root/queue2_t4a.sh` lanzó sub053_1 automáticamente a las 09:21 UTC.
- sub053_1 abortado en la época 7 por orden del orquestador; rsna-g detenido.
