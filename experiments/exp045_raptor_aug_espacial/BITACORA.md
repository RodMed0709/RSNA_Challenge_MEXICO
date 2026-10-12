# exp045_raptor_aug_espacial — bitácora

- 2026-10-12 · claim creado; rama exp045_raptor_aug_espacial desde exp031_raptor_etiquetas_consenso (ca4f9b3). SPEC + notebook.
- Test unitario en GPU (rsna-h): espejo p=1 vs `_mirror_tri` del TTA nartaa max|Δ| 5.6e-5 (interpolación de grid_sample identidad); afín idéntico entre las K ventanas (Δ 0); eval == normalización del padre. Coste aug ≈ 7.6 ms/batch.
- 2026-10-12 02:18 smoke OK (1 época, 64/48): pseudo L2 0.510. Primer intento falló por falta de `RSNA_USER` (export RSNA_USER=rodrigo).
- 2026-10-12 02:21 run fold 0 lanzado en rsna-h (PID 1349).
- 2026-10-12 04:55 fold 0 terminado: pseudo L2 0.8899 (ep 15), SWA 0.8895, gold 0.9202, 2.51 GPU-h → worked.
- 2026-10-12 04:56 all-data lanzado en rsna-h (PID 4917), SWA_EPOCHS [13, 14, 15].
- 2026-10-12 07:45 all-data terminado: SWA [13, 14, 15], gold 0.9186, 2.81 GPU-h → swa_alldata.pt en results/ del worktree del pod.
