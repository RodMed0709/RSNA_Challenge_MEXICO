# exp033_raptor_24epocas — bitácora

- 2026-10-11 · rsna-g (RTX PRO 4500 32 GB, corpus en `/root/corpus96_384`). Claim creado. Padre exp029 corriendo en rsna-d2.
- Pod: worktree `wt_exp033_raptor_24epocas`, `export RSNA_USER=rodrigo`, symlink `results/exp029_raptor_entreno`; lanzador `/root/run033.sh` con `setsid nohup`.
- Smoke (corpus completo, 64/48, 1 época): pasa → `metrics_smoke.json`. Al lanzar, el pod tenía otro proceso ajeno en GPU (`t4_fast.py`, ~6 GB); no se tocó.
- 19:31 UTC run real fold 0 (log `results/exp033_raptor_24epocas/run_20261011_193119.log`): ~504 s train + ~62 s eval por época; ep0 pseudo L2 0.7059 · v1 0.6878 · gold 0.7073.
- Fin: mejor época 14 (pseudo L2 0.8695), luego baja hasta 0.8651 en ep 23; gold 0.9195; 3.79 GPU-h → `didnt`.
