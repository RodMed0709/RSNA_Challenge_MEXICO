# exp000_preparacion

| Campo | Valor |
|---|---|
| Autor | rodrigo |
| Padre | — |
| Estado | worked |

Todo lo que se corrió una sola vez antes de entrenar. Los notebooks van en el orden en que se ejecutaron.
No se vuelven a correr salvo que se quiera una versión nueva (cache v2, etiquetas v2, split v2), y eso sería otro experimento.

| # | Notebook | Qué produce | Resultado |
|---|---|---|---|
| 1 | `01_explorar_datos.ipynb` | Revisión de CSV, series y DICOM | 4,407 estudios, 24,371 series, 58 gold |
| 2 | `02_crear_cache.ipynb` | `/workspace/data/cache/v1_256/` (un `.npy` uint8 por serie + `index.csv`) | 24,371 series, 0 errores, 59 GB, ~15 min |
| 3 | `03_etiquetas_qwen.ipynb` | `/workspace/data/labels/report_labels_v1.csv` (Qwen2.5-7B lee los informes) | Acuerdo con gold ~0.80; flojas MCL, Effusion, Synovitis |
| 4 | `04_split_v1.ipynb` | `splits/split_v1_seed42.csv` + `.json` | 5 folds de 865–877, gold en fold -1 (ver `docs/07_SPLIT.md`) |

Salidas de ejecución en `results/exp000_preparacion/`.

**Qué aprendimos:** no hay forma de identificar pacientes (PatientID distinto por estudio); el idioma del informe
aproxima el hospital; en las etiquetas v1 "no mencionado" quedó como 0.5 y probablemente debería ser 0.
