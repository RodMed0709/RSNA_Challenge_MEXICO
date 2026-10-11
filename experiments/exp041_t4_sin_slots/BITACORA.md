# Bitácora exp041_t4_sin_slots

- 2026-10-11 · Worktree `RSNA_wt041` desde `exp027_t4_rapido`. Claim vía rsna-c2
  (`/workspace/users/.claims/exp041_t4_sin_slots`; rsna-e no responde por ssh).
- Cómo quitó exp040 un slot: "una serie faltante se simula igual que la representa el corpus: cortes a cero y
  máscara 0". Aquí: lector `study_lut` del notebook con `SLOTS[:3]` (no lista ni lee los DICOM de Cor no fluido
  ni Axial) y relleno con ceros hasta 96 en el consumidor; K = 64 (centros válidos de 66 cortes). El arnés
  comprueba que el volumen de 3 slots es idéntico a los primeros 66 cortes del de 5 (estudio de calentamiento).
- Estudios (`scratchpad/kaggle_exp041/exp041_studies.csv`): 150 main + 2 warmup de exp027, `cold5` y `cold3`
  = 2 × 150 de fold 0 nuevos (semilla 2, excluidos los de exp026/exp027).
- Smoke en rsna-f (PRO 4500, `/kaggle` simulado, 3 estudios, junto a un entrenamiento ajeno, ~2 min):
  todas las variantes corren; identidad 3 slots == primeros 66 cortes: True.
- Push del bench `medellinrodrigo/exp041-t4-sin-slots` v1: RUNNING sin cola.
- Kernel v1 COMPLETE, 1 152 s. Salidas en `scratchpad/kaggle_exp041/bench_out/` (no van a git);
  `analyze.py` → `results/exp041_t4_sin_slots/metrics.json`.
- Kernels de envío (sin enviar) en `scratchpad/kaggle_exp041/`: `submit_C` (AUC) y `submit_E1_94` (= Ch,
  eficiencia); también `submit_C3`, `submit_C3h`, `submit_C3t` (dominadas, sin probar en pod).
- Resultado en SPEC §3.
