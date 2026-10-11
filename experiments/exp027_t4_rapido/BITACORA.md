# Bitácora exp027_t4_rapido

- 2026-10-11 · Worktree `RSNA_wt027` desde `exp026_t4_variantes`. Claim en rsna-e
  (`/workspace/users/.claims/exp027_t4_rapido`).
- Modelo 224: `nartaa/rsna-knee-0945-efficient-224crop` (LB 0.945 según el README del dataset; el 384 con
  TTA espejo es 0.949). Receta de su celda: mismo volumen 96×384 (study_lut), ventana = recorte central 0.8
  (307 px) → bilinear 224 → ImageNet, `coatnet_rmlp_2_rw_224`, 94 ventanas en un lote, autocast, sin TTA.
  El arnés importa las dos celdas públicas; la preparación es la del notebook 0949 para todas las variantes.
- `model.load` ya es una vez por GPU en el notebook (un hilo consumidor por GPU): la parte "modelo cargado
  una vez" de A no cambia nada; A = solo `cudnn.benchmark = False`.
- `reduce-overhead` descartado: CUDA graphs por hilo; D/E2 usan `max-autotune-no-cudagraphs`, compilados y
  calentados en el hilo principal antes de lanzar los consumidores.
- Estudios (`scratchpad/kaggle_exp027/exp027_studies.csv`): 150 main + 2 warmup de exp026 y 150 de fold 0
  nuevos (rol `cold`, semilla 1, sin etiquetas en uso).
- Smoke en rsna-g (PRO 4500, 1 GPU, `/kaggle` simulado con symlinks, 3 estudios): todas las variantes
  corren; F falla limpio (no hay torch_tensorrt ni onnxruntime en el venv). Compilar max-autotune: 246 s
  (D) y 193 s (E2) con 48 CPU. Todo lo que corrió después de compilar salió más lento (prep sola 1.5 vs
  ~0.7 s/estudio) → compiladas al final + `shutdown_compile_workers()` + control E1b; variante fría = la 224
  sin compilar más rápida. Bug del smoke corregido: la corrida fría heredaba `cudnn.benchmark=True` de Rb.
  Segundo smoke cortado a mitad: alguien lanzó exp033 en rsna-g (no compito por la GPU).
- Push del bench `medellinrodrigo/exp027-t4-rapido` v1: RUNNING sin cola.
