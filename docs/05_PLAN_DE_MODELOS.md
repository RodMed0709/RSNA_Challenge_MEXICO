# Plan de modelos

La idea es subir un escalón a la vez. Cada etapa deja algo reusable y una medición clara; no saltamos al modelo grande antes de comprobar que los datos y etiquetas funcionan.

## 0. Explorar

Corremos `experiments/exp001_preparacion/01_explorar_datos.ipynb` para entender los 4,407 estudios, los 58 con etiqueta experta, las series por plano y algunos DICOM. Aquí buscamos errores de ruta, estudios incompletos y diferencias raras de intensidad.

## 1. Crear cache

Rodrigo corre una sola vez `experiments/exp001_preparacion/02_crear_cache.ipynb`. Decodificar DICOM durante cada entrenamiento sería lento y competiría por el disco compartido. El cache `v1_256` normaliza y redimensiona cada corte, pero conserva todas las series y todos los cortes para cambiar el muestreo después.

## 2. Sacar pseudo-etiquetas de los informes

`experiments/exp001_preparacion/03_etiquetas_qwen.ipynb` lee los 4,407 informes con un Qwen Instruct local y guarda 12 valores por estudio: presente, ausente o incierto. Es reanudable porque escribe resultados conforme avanza. Al final compara contra los 58 gold por etiqueta. Revisamos especialmente etiquetas con AUC o acuerdo bajo; un informe clínico y la etiqueta de expertos no siempre describen exactamente lo mismo.

## 3. Baseline DINOv2-small

exp002 (`experiments/exp002_dinov2s_ref/`) toma hasta cinco series por estudio y muestrea cortes de cada una. DINOv2-small extrae una representación por corte; dos capas de atención juntan primero cortes y luego series. Agregamos el plano y el tipo de secuencia como embeddings. El primer objetivo es cerrar un fold completo, guardar OOF y tener una AUC reproducible, no exprimir el leaderboard.

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

Elegimos checkpoints que mejoren OOF y que no sean copias casi idénticas. `inferencia_kaggle.ipynb` lee DICOM de test directamente, promedia probabilidades y respeta el orden de `sample_submission.csv`. Antes de enviar, corremos el notebook completo sin internet y confirmamos tiempo, rutas, pesos incluidos, número de filas, columnas y ausencia de NaN.

## Dónde vamos (24 sep)

| Paso | Estado | Resultado |
|---|---|---|
| Datos, cache `v1_256`, pseudo-etiquetas v1 | listo | 4,407 estudios, 59 GB de cache, acuerdo Qwen vs gold ~0.80 |
| Split canónico `split_v1_seed42` | listo | ver `docs/07_SPLIT.md` |
| exp002 referencia sobre split_v1 | corriendo 5 folds | fold 0: pseudo 0.827, gold 0.777 (igual que la prueba previa al split). **Fold 1: pseudo 0.738, gold 0.690** — la receta es inestable al descongelar el backbone |

## Backlog para el equipo (octubre)

El nivel público está en ~0.94 y nosotros en 0.777 de gold: hay que cambiar datos, entrada y cabeza, no afinar.
Detalle de cada idea en `docs/08_LO_QUE_HACEN_OTROS.md`. Nadie tiene número reservado: quien tome una idea corre
`nuevo_experimento(...)`, escribe su SPEC y la marca en el grupo. Padre = referencia actual de `docs/EXPERIMENTOS.md`.

| Prioridad | Idea | Tipo de cambio | Por qué |
|---|---|---|---|
| 0 | Enviar a Kaggle un fork del notebook público 0.943 | Envío, no experimento | Valida el flujo de envío y nos pone en el leaderboard |
| 0 | Inferencia propia cronometrada con exp002 (`inferencia_kaggle.ipynb`) | Envío | Rutas offline, tiempo real, formato |
| 1 | Cache v2: recorte 130 mm, 336 px, rodillas alineadas, 6 slots por tipo de secuencia | Datos (nuevo cache) | Mayor palanca: detalle fino y series correctas |
| 1 | Etiquetas v2: Qwen3-8B, "no mencionado" bajo (~0.2) con peso bajo, confianza por etiqueta | Etiquetas | Menos ruido; hoy "no mencionado" = 0.5 |
| 1 | Cabeza con atención por etiqueta + lr cabeza 1e-3 | Modelo (celda 3) | Cada hallazgo mira su plano; también ataca el fold inestable |
| 2 | Entrada 2.5D (3 cortes vecinos como RGB) | Datos/modelo | Contexto 3D barato |
| 2 | Congelar la primera mitad de DINOv2 + warmup después de descongelar | Entrenamiento | Estabilidad (fold 1 cayó a 0.738) |
| 2 | Pooling por etiqueta en test (max / top2 / media) | Evaluación | Hallazgos focales vs difusos |
| 3 | CoAtNet `coatnet_rmlp_2_rw_384` en lugar de DINOv2 | Modelo | Backbone del mejor notebook público |
| 3 | Promedio de checkpoints top-3 y mezcla por ranking | Entrenamiento/ensemble | Gratis en precisión |
| 4 | Ensemble final: 5 folds × 2 semillas × 2 backbones, TTA | Final (desde 8 oct) | Última milla |

Cada época tarda ~11 min con la receta actual (~90 min por fold). Screening siempre con fold 0.
