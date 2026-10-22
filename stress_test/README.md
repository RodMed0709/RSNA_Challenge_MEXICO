# Stress Test — benchmark privado de generalización

Estado: borrador de diseño pendiente de confirmación; todavía no hay un benchmark materializado ni un modelo seleccionado.

## Para qué existe

`stress_test` medirá si los modelos mantienen su ordenamiento de hallazgos cuando cambian el dominio de adquisición o se deterioran de forma plausible las imágenes. También probará —sin asumirlo— si esas mediciones ayudan a anticipar el leaderboard público.

No es un dataset de Kaggle ni una copia de DICOM. Las imágenes fuente permanecen en `/workspace/data/raw/`; las perturbaciones se generan al vuelo. Los artefactos del benchmark guardan IDs, grupos, condición, semilla y parámetros, además de métricas agregadas. No se publican imágenes, informes ni etiquetas.

## Principio contra fuga y sobreajuste

Los casos difíciles se identifican usando predicciones out-of-fold, nunca predicciones de un modelo sobre imágenes con las que se entrenó. El benchmark separará descubrimiento, desarrollo y lockbox. El lockbox es una barrera de proceso (no un enclave contra el dueño del volumen): se congela antes de evaluar y sus resultados no deben guiar ajustes repetidos.

La lista de ~70 envíos históricos sirve para un análisis retrospectivo de transferencia, no convierte esos mismos modelos en una evaluación futura independiente. Se agrupan por familia/linaje y fecha; una métrica de estrés solo podrá orientar decisiones si predice scores de leaderboard fuera de familia y en una validación temporal.

## Archivos

- `SPEC.md`: definición, límites de datos, paneles y gates.
- `TRANSFER.md`: protocolo para comprobar relación con leaderboard sin sobreajustar.
- `RESULTS.md`: registro de cobertura y resultados, se llena después del smoke y la auditoría histórica.

## Uso previsto

1. Auditar qué predicciones/checkpoints OOF y qué scores de Kaggle están disponibles; no asumir que los ~70 envíos son 70 modelos independientes.
2. Congelar el manifiesto `stress_test_v1` y sus transformaciones.
3. Ejecutar el mismo benchmark sobre los modelos comparables.
4. Evaluar si la métrica predice el leaderboard y después usarla para el screen de GroupDRO.

El estado actual del proyecto prohíbe editar `split_v1_seed42`; cualquier validación por dominios será un artefacto separado y no reemplazará el split canónico.
