# Plan de modelos

La idea es subir un escalón a la vez. Cada etapa deja algo reusable y una medición clara; no saltamos al modelo grande antes de comprobar que los datos y etiquetas funcionan.

## 0. Explorar

Corremos `00_explorar_datos.ipynb` para entender los 4,407 estudios, los 58 con etiqueta experta, las series por plano y algunos DICOM. Aquí buscamos errores de ruta, estudios incompletos y diferencias raras de intensidad.

## 1. Crear cache

Rodrigo corre una sola vez `01_crear_cache.ipynb`. Decodificar DICOM durante cada entrenamiento sería lento y competiría por el disco compartido. El cache `v1_256` normaliza y redimensiona cada corte, pero conserva todas las series y todos los cortes para cambiar el muestreo después.

## 2. Sacar pseudo-etiquetas de los informes

`02_etiquetas_con_qwen.ipynb` lee los 4,407 informes con un Qwen Instruct local y guarda 12 valores por estudio: presente, ausente o incierto. Es reanudable porque escribe resultados conforme avanza. Al final compara contra los 58 gold por etiqueta. Revisamos especialmente etiquetas con AUC o acuerdo bajo; un informe clínico y la etiqueta de expertos no siempre describen exactamente lo mismo.

## 3. Baseline DINOv2-small

`03_baseline_dinov2.ipynb` toma hasta cinco series por estudio y muestrea cortes de cada una. DINOv2-small extrae una representación por corte; dos capas de atención juntan primero cortes y luego series. Agregamos el plano y el tipo de secuencia como embeddings. El primer objetivo es cerrar un fold completo, guardar OOF y tener una AUC reproducible, no exprimir el leaderboard.

Primero se puede congelar el backbone y después afinarlo. Usamos BCE con máscara para soportar NaN y aceptar `0.5` como etiqueta suave. No hacemos flip horizontal porque puede cambiar medial por lateral.

## 4. Mejorar de forma controlada

Cuando el baseline sea estable probamos una sola variable por run:

- más cortes o una resolución mayor;
- más series y otra prioridad de secuencias;
- backbones más grandes de DINOv2 o modelos `timm`;
- mejor agregación de cortes/series;
- ponderar gold y pseudo-etiquetas, o filtrar incertidumbre;
- cinco folds y semillas adicionales.

El costo también importa: Kaggle permite 9 horas y el test oculto es grande. Una mejora pequeña que duplica el tiempo puede no servir.

## 5. Ensemble e inferencia Kaggle

Elegimos checkpoints que mejoren OOF y que no sean copias casi idénticas. `04_inferencia_kaggle.ipynb` lee DICOM de test directamente, promedia probabilidades y respeta el orden de `sample_submission.csv`. Antes de enviar, corremos el notebook completo sin internet y confirmamos tiempo, rutas, pesos incluidos, número de filas, columnas y ausencia de NaN.
