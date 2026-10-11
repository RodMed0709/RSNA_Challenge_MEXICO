# exp029_raptor_entreno — bitácora

- 2026-10-11 17:00 UTC · rsna-d2 (48 vCPU visibles, RTX PRO 4500 32 GB, disco local 200 GB). Claim creado.
- Corpus: `build_corpus96.py` → `/root/corpus96_384`. Prueba 96 estudios: 16 procesos = 3.0 est/s; con 32 procesos
  también ~3.1–3.3 est/s (limitado por el volumen de red, carga ~130) → ETA ~23 min para 4407.
- Corpus completo: 4407 estudios en **1317 s (21.9 min)**, 32 procesos, 0 errores, 0 estudios vacíos, 92.3 cortes
  válidos de 96 de media; `vols.npy` 62.4 GB (59 GiB). Copia a `/workspace/data/cache/raptor96_384/` (vols/masks/ids/
  build_meta) en **94 s**.
- Verificación del corpus: pesos publicados de nartaa (`..._d96_r384_swa.pt`, sin TTA, k=94, recorte 320) sobre
  nuestro corpus → **gold-58 = 0.9219** (metadatos del checkpoint: 0.9190). El corpus reproduce el de nartaa.
  (Solo comprobación de pipeline; gold no selecciona nada.)
- Gotcha: en el worktree del pod hace falta `export RSNA_USER=rodrigo` (si no, `current_user()` falla).
- Smoke (corpus pequeño gold + 150, 64 train / 28 val, 1 época): pasa todas las celdas; 8 pasos en 25 s; metrics.json OK
  (renombrado a `metrics_smoke.json`).
- 17:39 UTC run real fold 0: train 3477 · val 871 · 434 pasos/época · ~1.5 s/paso, VRAM ~19 GB.
