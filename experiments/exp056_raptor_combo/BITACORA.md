# exp056_raptor_combo — bitácora

- 2026-10-12 06:35 · claim creado; rama exp056_raptor_combo desde exp045_raptor_aug_espacial (72de635). SPEC + notebook con interruptores (todos apagados = exp045).
- 2026-10-12 06:51 · por pedido del orquestador: exp045 fold 1 lanzado en rsna-f (worktree del volumen wt_exp045_f1, detached en la rama exp045, PID 23502). Si el combo = exp045, esos folds son los del combo.
- 2026-10-12 07:05 · verdicts de la tanda: exp048 mixup 0.8845 (+0.0009) y exp052 etiqueta estocástica 0.8851 (+0.0015) → no entran. Esperando exp050 y sub046_1.
- 2026-10-12 07:12 · exp045 fold 2 lanzado en rsna-c2 (wt_exp045_f2, PID 21210). exp050 res384 didnt (+0.0019, ×1.46 coste) → no entra.
- 2026-10-12 07:30 · sub046_1 didnt (0.8850, +0.0014). Ningún componente entra → combo = exp045; fold 0 = el de exp045. exp045 fold 3 lanzado en rsna-i (wt_exp045_f3, PID 8374). Fold 4 → rsna-h tras el all-data de exp045.
- 2026-10-12 ~07:55 · orquestador: exp054 (L5) worked → combo = exp045 + consenso 5 fuentes (cambio de exp054). Notebook rehecho desde exp045 (sin interruptores); SPEC reescrita con criterio ≥ 0.8889 / gold ≥ 0.9002. Fold 0 → rsna-d2 tras sub045_2; folds 1–4 tras los de exp045.
- 2026-10-12 09:08 · smoke OK en rsna-d2 (5 fuentes, cobertura completa en train; acuerdo L5–L2 0.882). 09:09 fold 0 lanzado en rsna-d2 (/root/run_exp056_f0.sh), fin ~11:40.
- 2026-10-12 09:25 · lanzadores encadenados /root/chain_exp056_fK.sh en F (f1), C2 (f2), I (f3), H (f4): esperan a que termine exp045 fold K en ese pod y lanzan exp056 fold K.
- 2026-10-12 10:50 · all-data seed 42 (rsna-j, PID 348) y seed 7 (rsna-k, PID 218, worktree wt_exp056_seed7), SWA [13,14,15]; 5090 a 0.60 s/paso.
- 2026-10-12 11:41 · fold 0 terminado: pseudo L2 0.8857 (ep 15), SWA 0.8854, gold 0.9198, 2.51 GPU-h → didnt (umbral 0.8889). 11:42 parados folds 1–4 (F, C2, I, H) y all-data (j, k); paradas de pods automáticas.
- exp045 5 folds: OOF pseudo L2 0.8844, gold ensamble 5 folds 0.9244.
