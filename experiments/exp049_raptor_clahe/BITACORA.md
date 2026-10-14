# exp049_raptor_clahe — bitácora

- 2026-10-12 · SPEC y notebook preparados; claim creado.
- rsna-e2: worktree `wt_exp049`. `build_corpus_clahe.py --procs 16` (nice/ionice) → `/root/corpus_clahe` en 30 s de pared (corpus fuente en page cache); **CLAHE 66 ms por estudio** (96 cortes 384², 1 hilo) = costo extra de inferencia por estudio. Test previo (40 estudios): cortes no válidos siguen a cero; media/desv. casi iguales (114/78 → 110/76), los ceros dentro de la imagen desaparecen (8–19 % → 0 %).
- Lanzador `/root/t3_chain.sh` (setsid nohup, log `/root/t3_chain.log`): espera a que acabe el all-data seed 7 (sub031_2) y luego smoke → run real fold 0.
- 03:29 UTC run real fold 0 en rsna-e2 (log `results/exp049_raptor_clahe/run_20261012_032900.log`): 1.30 s/paso (434 pasos/época). Smoke OK (03:27).
- 05:58 fin: mejor época 14, pseudo L2 0.8864 (Δ +0.0028), gold 0.9083, 2.47 GPU-h → `didnt`. Pesos en el pod: `wt_exp049/results/exp049_raptor_clahe/{best,swa}_f0.pt`.
