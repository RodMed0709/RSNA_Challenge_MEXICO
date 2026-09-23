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

En cada época medimos `pseudo_auc` sobre el fold de pseudo-etiquetas y `gold_auc` sobre todos los estudios gold, que nunca entran en folds ni entrenamiento. `select_on` decide cuál de las dos métricas elige el mejor checkpoint; por defecto usamos `pseudo`.

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

## Dónde vamos (23 sep)

| Paso | Estado | Resultado |
|---|---|---|
| Datos en el volumen | listo | 4,407 estudios, 24,371 series, 585 GB en `/workspace/data/raw` |
| Cache `v1_256` | listo | 24,371 series, 0 errores, 59 GB, ~15 min con 14 workers |
| Pseudo-etiquetas `report_labels_v1` (Qwen2.5-7B) | listo | AUC vs gold ~0.80; flojas: Synovitis 0.66, PF OA 0.70, Effusion 0.73 |
| Baseline `rodrigo_dinov2s_baseline_v1` (lr 1e-4 para todo) | descartado | se cae al descongelar el backbone: gold 0.64 → 0.60 |
| Baseline `rodrigo_dinov2s_bblr1e5_v2` (backbone 1e-5, 8 épocas, fold 0) | referencia | pseudo_auc 0.828, gold_auc 0.781, ~24 min en RTX PRO 6000 |

Siguientes ejes para repartir: mejores pseudo-etiquetas (Qwen más grande / prompt con definiciones), más épocas y cortes, `dinov2-base`, los 5 folds, e inferencia Kaggle cronometrada.
