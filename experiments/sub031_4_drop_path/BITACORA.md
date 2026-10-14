# sub031_4_drop_path — bitácora

- 2026-10-12 · rsna-f (RTX PRO 4500 32 GB, corpus en `/root/corpus96_384`). Claim creado. Padre exp031 (`worked`, pseudo L2 0.8836).
- timm 1.0.30: `drop_path_rate` por defecto de coatnet_rmlp_2_rw_384 = 0.0 (0 módulos DropPath); con 0.2 → 39 DropPath, 0.009 → 0.200.
- Smoke OK → `metrics_smoke.json`. 01:40 UTC 12-oct run real fold 0 (log `results/sub031_4_drop_path/run_20261012_014010.log`, lanzador `/root/run031_4.sh`): ~509 s train + ~76 s eval por época; ep0 pseudo L2 0.7258. Fin: mejor ep 15, pseudo L2 0.8849, gold 0.9097, 2.55 GPU-h → `didnt`.
