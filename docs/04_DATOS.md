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
| `/workspace/users/<nombre>/` | Clon personal del repo. | Esa persona. |
| `/workspace/users/<nombre>/results/expNNN_tag/` | Pesos, predicciones y `metrics.json` de cada experimento. | Esa persona. |
| `/workspace/data/_logs/` | Logs de descarga, unzip y autoapagado. | Rodrigo. |

## CSV

- `train.csv`: `StudyInstanceUID`, `Report` y las 12 etiquetas. Solo 58 de 4,407 filas tienen etiqueta experta. Está en UTF-8: léelo con `encoding="utf-8"` (con latin-1 sale basura como "TÃ©cnica").
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

`experiments/exp001_preparacion/02_crear_cache.ipynb` produce:

```text
/workspace/data/cache/v1_256/<StudyInstanceUID>/<SeriesInstanceUID>.npy
/workspace/data/cache/v1_256/index.csv
```

Cada `.npy` es `uint8` con forma `[cortes, 256, 256]`; conserva todos los cortes. `index.csv` guarda estudio, serie, plano, flags de secuencia, número de cortes y ruta. El proceso se puede reanudar: si una serie ya existe, se salta. Las fallas quedan en `errors.csv` sin parar las demás.

## Etiquetas y split

- `/workspace/data/labels/report_labels_v1.csv`: pseudo-etiquetas de Qwen2.5-7B (1, 0, 0.5 incierto). Acuerdo con gold ~0.80;
  flojas en MCL, Effusion y Synovitis. Ojo: hoy "no mencionado en el informe" queda como 0.5 (`src/knee/labels.py`), probablemente debería ser 0.
- El split canónico está en `SPLITS/` y se explica en `docs/07_SPLIT.md`.
