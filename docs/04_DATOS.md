# Datos

## Volumen compartido

| Ruta | Contenido | Quién escribe |
|---|---|---|
| `/workspace/data/raw/` | Datos de Kaggle descomprimidos. | Nadie; solo lectura. |
| `/workspace/data/cache/<version>/` | Series preprocesadas por notebook 01. | Rodrigo. |
| `/workspace/data/labels/` | Pseudo-etiquetas de informes. | Rodrigo. |
| `/workspace/models/dinov2-small/` | Pesos locales de `facebook/dinov2-small`. | Rodrigo. |
| `/workspace/models/` | Otros pesos descargados. | Rodrigo. |
| `/workspace/venv/` | Ambiente compartido de Python. | Script de setup. |
| `/workspace/users/<nombre>/RSNA_Challenge/` | Clon personal del repo. | Esa persona. |
| `/workspace/users/<nombre>/runs/<run_name>/` | Pesos, logs y resultados personales. | Esa persona. |
| `/workspace/results/all_runs.csv` | Tabla compartida de experimentos. | Entrenamiento, al terminar. |

## CSV

- `train.csv`: `StudyInstanceUID`, `Report` y las 12 etiquetas. Solo 58 de 4,407 filas tienen etiqueta experta. Si el texto se ve raro, prueba primero UTF-8 y después `latin-1` o `cp1252`.
- `train_series.csv`: una fila por serie con estudio, serie, plano anatómico, sensibilidad a líquido y supresión de grasa.
- `test.csv`: lista de estudios de test; no contiene informes.
- `test_series.csv`: metadatos de las series de test.
- `sample_submission.csv`: orden exacto de estudios y columnas para entregar.

## DICOM

Cada archivo es un corte 2D:

```text
/workspace/data/raw/train_series/<StudyInstanceUID>/<SeriesInstanceUID>/<SOPInstanceUID>.dcm
/workspace/data/raw/test_series/<StudyInstanceUID>/<SeriesInstanceUID>/<SOPInstanceUID>.dcm
```

Hay DICOM sin comprimir, JPEG Lossless y JPEG 2000. `src/knee/dicom.py` ordena cortes por posición física, aplica slope/intercept, corrige `MONOCHROME1`, iguala tamaños y normaliza percentiles 1–99.

## Cache

`notebooks/01_crear_cache.ipynb` produce:

```text
/workspace/data/cache/v1_256/<StudyInstanceUID>/<SeriesInstanceUID>.npy
/workspace/data/cache/v1_256/index.csv
```

Cada `.npy` es `uint8` con forma `[cortes, 256, 256]`; conserva todos los cortes. `index.csv` guarda estudio, serie, plano, flags de secuencia, número de cortes y ruta. El proceso se puede reanudar: si una serie ya existe, se salta. Las fallas quedan en `errors.csv` sin parar las demás.
