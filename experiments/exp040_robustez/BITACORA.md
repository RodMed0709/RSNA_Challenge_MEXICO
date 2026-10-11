# exp040_robustez — bitácora

- 2026-10-11 20:32 UTC · rsna-e (RTX PRO 4500 32 GB, disco local 40 GB, sin otros runs). Claim creado.
- Modelo: `wt_exp029/results/exp029_raptor_entreno/best_f0.pt` (época 15, pseudo L2 0.8682; SWA top-3 0.8679 → se usa best).
- Smoke (20 estudios, 12 condiciones): todas corren, ~1 s por condición. La referencia limpia reproduce las predicciones
  OOF de exp029 con diferencia máxima 0.0022 (bf16, tamaño de chunk distinto).
- Metadatos (`meta_subgrupos.py`): fabricante y campo en meta_v2; `PatientSex` en DICOM (M 453, F 384, O 39, resto vacío);
  **`PatientAge` vacía en todos** (anonimizada) → sin subgrupo de edad.
- Run real: cache de los 929 estudios en `/root/exp040_cache.npy` (13.2 GB, ~7 min desde el volumen a ~30 MB/s).
- 32 pasadas: ~61 s cada una a k24 (k48 122 s, todas las ventanas 237 s); total 0.59 GPU-h. `analisis_robustez.py` (~1 min,
  CPU): referencia pseudo 0.8682 (= exp029). Resultados en SPEC §3. Ojo: `pgrep -f infer_robustez` también encuentra la
  shell padre de setsid; comprobar fin por el log.
