# sub031_1_consenso_jev — bitácora

- 2026-10-11 · rsna-g (RTX PRO 4500 32 GB, corpus en `/root/corpus96_384`). Claim creado. Padre exp031 (`worked`, pseudo L2 0.8836).
- JEV `FINAL2_labels_scores.csv` (ensemble de texto): 4407 estudios, 0 NaN, blanda (52 884 valores), cubre 4348/4348 del pool, media 0.148; acuerdo binarizado en pool JEV–L2 0.879, JEV–Raptor 0.909. Se usa FINAL2 (JEV) y no HYBRID (≈95 % L2/L3, ya en el consenso).
- 01:36 UTC 12-oct run real fold 0 (log `results/sub031_1_consenso_jev/run_20261012_013655.log`, lanzador `/root/run031_1.sh`): ~504 s train + ~74 s eval por época; ep0 pseudo L2 0.7224. Smoke previo OK (consenso 5 fuentes: 41 724/41 724 celdas con las 5; media objetivo 0.271). Fin: mejor ep 11, pseudo L2 0.8846, gold 0.9107, 2.53 GPU-h → `didnt`.
