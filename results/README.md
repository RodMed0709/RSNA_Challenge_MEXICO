# Registro de resultados

El archivo real vive en `/workspace/results/all_runs.csv` para que todos vean los experimentos terminados. `train_fold` agrega una fila automáticamente al finalizar cada fold.

Columnas base: `timestamp`, `user`, `run_name`, `fold`, `backbone`, `img_size`, `n_slices`, `epochs`, `lr`, `val_auc_mean`, una columna `auc_<label>` por hallazgo y `notes`.

Este directorio del repo solo documenta el formato. No copies aquí pesos ni resultados grandes.
