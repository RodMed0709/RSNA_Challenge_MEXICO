# Registro de resultados

El archivo real vive en `/workspace/results/all_runs.csv` para que todos vean los experimentos terminados. `train_fold` agrega una fila automáticamente al finalizar cada fold.

Columnas base: `timestamp`, `user`, `run_name`, `fold`, `backbone`, `img_size`, `n_slices`, `epochs`, `lr`, `pseudo_auc`, `gold_auc`, `select_on`, `val_auc_mean`, una columna `auc_<label>` por hallazgo y `notes`.

`pseudo_auc` mide el fold de pseudo-etiquetas y `gold_auc` mide todos los estudios gold separados del entrenamiento. `val_auc_mean` contiene la métrica elegida por `select_on`; cuando hay gold, las columnas `auc_<label>` también corresponden a gold.

Este directorio del repo solo documenta el formato. No copies aquí pesos ni resultados grandes.
