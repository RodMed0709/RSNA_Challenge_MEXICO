# exp050_raptor_res384 — bitácora

- 2026-10-12 · SPEC y notebook preparados; claim creado.
- rsna-c2: worktree `wt_exp050`, lanzador `/root/t3_chain.sh` (espera a que acabe exp044, luego smoke → run real fold 0; log `/root/t3_chain.log`).
- Smoke OK (03:26, 33 s train/20 s eval vs 25/15 s de exp049 a 320). 03:28 UTC run real fold 0 en rsna-c2 (log `results/exp050_raptor_res384/run_20261012_032841.log`): **1.87 s/paso** (vs ~1.14–1.30 a 320 ⇒ ~1.45×), **VRAM 26.7 GB** a bs 8 × k 12 (cabe sin cambiar batch).
- 07:07 fin: mejor época 15, pseudo L2 0.8856 (Δ +0.0019), gold 0.9172, 3.63 GPU-h → `didnt`. Pesos en el volumen: `wt_exp050/results/exp050_raptor_res384/{best,swa}_f0.pt`.
