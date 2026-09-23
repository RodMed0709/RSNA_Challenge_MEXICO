# RSNA Knee Abnormality Detection

Competencia de Kaggle para predecir 12 hallazgos en cada resonancia de rodilla. El modelo recibe varias series DICOM por estudio y produce 12 probabilidades. La métrica es el promedio de las 12 AUC ROC.

## Inicio rápido

1. Crea un pod con el volumen `RSNA-KNEE` montado en `/workspace`.
2. Clona este repo dentro de `/workspace/users/<tu_nombre>/`.
3. Define tu usuario: `export RSNA_USER=<tu_nombre>`.
4. Corre `bash scripts/setup_pod.sh` y elige el kernel `Python (rsna)` en JupyterLab.
5. Abre y corre `notebooks/00_explorar_datos.ipynb`.

## Mapa

- `notebooks/`: flujo oficial de exploración, cache, etiquetas, entrenamiento e inferencia.
- `src/knee/`: lectura DICOM, datos, modelo, entrenamiento e inferencia.
- `scripts/`: preparación del pod y ejecución en segundo plano.
- `experiments/`: copias personales de notebooks.
- `docs/`: guía práctica del equipo.
- `tests/`: smoke test local.

Empieza en [docs/00_EMPIEZA_AQUI.md](docs/00_EMPIEZA_AQUI.md). Después revisa [el challenge](docs/01_EL_CHALLENGE.md), [las reglas](docs/02_REGLAS_DEL_EQUIPO.md), [RunPod](docs/03_RUNPOD.md), [los datos](docs/04_DATOS.md) y [el plan de modelos](docs/05_PLAN_DE_MODELOS.md).
