# Plan de modelos

La idea es subir un escalón a la vez. Cada etapa deja algo reusable y una medición clara; no saltamos al modelo grande antes de comprobar que los datos y etiquetas funcionan.

## 0. Explorar

Corremos `experiments/exp000_preparacion/01_explorar_datos.ipynb` para entender los 4,407 estudios, los 58 con etiqueta experta, las series por plano y algunos DICOM. Aquí buscamos errores de ruta, estudios incompletos y diferencias raras de intensidad.

## 1. Crear cache

Rodrigo corre una sola vez `experiments/exp000_preparacion/02_crear_cache.ipynb`. Decodificar DICOM durante cada entrenamiento sería lento y competiría por el disco compartido. El cache `v1_256` normaliza y redimensiona cada corte, pero conserva todas las series y todos los cortes para cambiar el muestreo después.

## 2. Sacar pseudo-etiquetas de los informes

`experiments/exp000_preparacion/03_etiquetas_qwen.ipynb` lee los 4,407 informes con un Qwen Instruct local y guarda 12 valores por estudio: presente, ausente o incierto. Es reanudable porque escribe resultados conforme avanza. Al final compara contra los 58 gold por etiqueta. Revisamos especialmente etiquetas con AUC o acuerdo bajo; un informe clínico y la etiqueta de expertos no siempre describen exactamente lo mismo.

## 3. Baseline DINOv2-small

exp003 (`experiments/exp003_ref_split_v1/`) toma hasta cinco series por estudio y muestrea cortes de cada una. DINOv2-small extrae una representación por corte; dos capas de atención juntan primero cortes y luego series. Agregamos el plano y el tipo de secuencia como embeddings. El primer objetivo es cerrar un fold completo, guardar OOF y tener una AUC reproducible, no exprimir el leaderboard.

Primero se puede congelar el backbone y después afinarlo. Usamos BCE con máscara para soportar NaN y aceptar `0.5` como etiqueta suave. El flip de una sola imagen puede confundir medial con lateral; un espejo del estudio completo (todas las series juntas) sí es válido y está en el roadmap.

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

Elegimos checkpoints que mejoren OOF y que no sean copias casi idénticas. `kaggle/inferencia_kaggle.ipynb` lee DICOM de test directamente, promedia probabilidades y respeta el orden de `sample_submission.csv`. Antes de enviar, corremos el notebook completo sin internet y confirmamos tiempo, rutas, pesos incluidos, número de filas, columnas y ausencia de NaN.

## Dónde vamos (24 sep)

| Paso | Estado | Resultado |
|---|---|---|
| Datos, cache `v1_256`, pseudo-etiquetas v1 | listo | 4,407 estudios, 59 GB de cache, acuerdo Qwen vs gold ~0.80 |
| Split canónico `split_v1_seed42` | listo | ver `docs/07_SPLIT.md` |
| exp001 / exp002 (split viejo) | cerrados | exp002: pseudo 0.828, gold 0.781 |
| exp003 referencia sobre split_v1 | corriendo 5 folds | fold 0: pseudo 0.827, gold 0.777 (reproduce exp002). **Fold 1: pseudo 0.738, gold 0.690** — la receta es inestable al descongelar el backbone |

## Roadmap priorizado (un experimento = una variable, padre exp003)

| Prioridad | Idea | Por qué | Costo |
|---|---|---|---|
| 1 | Warmup después de descongelar (el warmup hoy cae en la época congelada y el backbone arranca con lr completo) | Explica la caída del fold 1 tras la época 1 | igual |
| 1 | Pseudo-etiquetas v2: "no mencionado" = 0 en vez de 0.5, prompt con definiciones, Qwen más grande | Probablemente la mayor fuente de ruido; +0.02–0.04 esperado en gold | GPU para relabel |
| 1 | Inferencia Kaggle cronometrada con exp003 | Obligatorio antes de todo lo demás: tiempo, rutas, offline | 1 envío |
| 2 | Selección de series que cubra los 3 planos | Hoy `head(5)` puede dejar fuera un plano | igual |
| 2 | Espejo del estudio completo (augment + TTA) | Rodilla derecha ↔ izquierda sigue siendo válida | igual |
| 2 | 24 cortes o 256 px | Más detalle | +30–60 % |
| 3 | `dinov2-base` | Backbone más grande | ×3 |
| 3 | Embedding posicional de cortes + 1 capa Transformer | El pooling actual ignora el orden de cortes | poco |
| 3 | EMA / 12 épocas | Estabilidad | +50 % |
| 4 | Pérdidas robustas al ruido, ensemble final con 2 semillas | Última semana | — |

Toda época tarda ~11 min con la receta actual (~90 min por fold). Screening siempre con fold 0.
