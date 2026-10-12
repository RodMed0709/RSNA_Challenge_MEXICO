# exp052_raptor_etiqueta_estocastica — bitácora

- 2026-10-12 · SPEC + notebook (copia de exp031, solo celdas de la variable). Claim en /workspace/users/.claims. Pod rsna-g ocupado por sub031_1 (ep3/16 a las 02:13 UTC); smoke y run en cuanto termine.
- Pod rsna-g: worktree `wt_exp052_raptor_etiqueta_estocastica`, symlink `results/exp031_raptor_etiquetas_consenso` → wt_exp031; cola `/root/queue_t4a.sh` (setsid nohup; espera sub031_1, smoke052, smoke053, real052, real053; log `/root/queue_t4a.log`).
- 04:11 UTC smoke (64/48, 1 época) rc=0: 4/4 fuentes en los 64 estudios; sorteo de fuentes en la época: raptor 14 · L2 14 · L3 17 · L4 19; pos_weight = regla del padre sobre consenso.
- 04:14 UTC run real fold 0 (log `results/exp052_raptor_etiqueta_estocastica/run_20261012_041354.log`).
- 06:47 UTC fin (rc=0): mejor época 15, pseudo L2 0.8851, gold 0.9150, 2.53 GPU-h → `didnt` (Δ +0.0014). Varas extra (best): Raptor 0.9003 · L3 0.8720 · L4 0.8847 · v1 0.8392. Pesos en el pod `wt_exp052_raptor_etiqueta_estocastica/results/exp052_raptor_etiqueta_estocastica/{best_f0,swa_f0}.pt`.
