# exp046_raptor_aug_intensidad — bitácora

- 2026-10-12 · claim creado; rama exp046_raptor_aug_intensidad desde exp031_raptor_etiquetas_consenso (ca4f9b3). SPEC + notebook.
- Test unitario en GPU: eval == normalización del padre; parámetros neutros = identidad (Δ 1.5e-8); todos los componentes activos dan valores finitos en [0,1]. Coste aug ≈ 7.3 ms/batch.
- 2026-10-12 02:18 smoke OK (1 época, 64/48): pseudo L2 0.493. Primer intento falló por falta de `RSNA_USER`.
- 2026-10-12 02:21 run fold 0 lanzado en rsna-i (PID 1184).
