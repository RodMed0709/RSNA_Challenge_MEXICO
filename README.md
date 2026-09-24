# RSNA Knee Abnormality Detection

Competencia de Kaggle para predecir 12 hallazgos en cada resonancia de rodilla. El modelo recibe varias series por estudio
y produce 12 probabilidades. La métrica es el promedio de las 12 AUC ROC.

Trabajamos con experimentos numerados y SDD: cada experimento tiene una SPEC antes de programarse y un verdict al terminar.
Reglas completas en [CLAUDE.md](CLAUDE.md).

## Mapa

- `docs/`: guía del equipo. Empieza en [docs/00_EMPIEZA_AQUI.md](docs/00_EMPIEZA_AQUI.md).
- `docs/EXPERIMENTOS.md`: registro global y referencia actual.
- `experiments/`: exp000 (preparación de datos) y luego exp001, exp002… en orden; cada uno con SPEC + notebook.
- `results/expNNN_tag/`: pesos y predicciones (solo `metrics.json` va a git).
- `splits/`: split canónico `split_v1_seed42`.
- `src/knee/`: código compartido (cache, dataset, modelo, métricas, experimentos).
- `kaggle/`: inferencia para enviar. La preparación de datos es `experiments/exp000_preparacion/`.
- `scripts/`: preparar el pod y correr notebooks en segundo plano.
