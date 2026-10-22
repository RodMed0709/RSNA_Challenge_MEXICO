# Protocolo de transferencia: stress test → leaderboard

## Pregunta

¿El desempeño OOF en dominios/casos difíciles y bajo estrés de adquisición ordena modelos de forma parecida a sus AUC públicas de Kaggle, fuera de la familia de modelos que originó el benchmark?

## Cohorte histórica

Una fila por modelo/versión realmente distinta. Reenvíos idénticos se agrupan; variantes de una misma familia se mantienen relacionadas. Para cada entrada registrar `experiment_id`, `model_family`, `parent`, `submission_ref`, fecha, public score, si cambió el artefacto, OOF disponible, checkpoint disponible y condiciones de estrés completadas.

No usar el CSV de submission (tiene predicciones del test sin etiquetas) como si fuera una predicción OOF etiquetada. No inferir AUC del test a partir de sus valores.

La cantidad bruta de envíos (~70) no es el tamaño muestral: copias/reenvíos, cambios mínimos y descendientes del mismo checkpoint se agrupan en familias. La auditoría debe informar tanto filas como familias independientes. Con menos de 8 familias evaluables o menos de 3 cortes cronológicos útiles, la correlación será descriptiva y no habilitará una regla de selección.

## Validación del vínculo

1. Medir correlación Spearman y Kendall entre cada métrica OOF de stress test y public AUC.
2. Hacer validación `leave-one-family-out`: seleccionar/calibrar cualquier combinación de métricas sin la familia evaluada.
3. Hacer validación temporal: calibrar con envíos anteriores y evaluar los posteriores, respetando fechas.
4. Bootstrap por familias de modelos, no por filas individuales; reportar intervalos y número de familias independientes.
5. Contrastar baseline limpio (pseudo OOF) contra stress-only y combinado. La pregunta es si stress añade señal predictiva al clean score, no si puede memorizar el leaderboard.
6. Considerar los scores Kaggle redondeados a 0.001 y su ruido; diferencias públicas menores que ~0.003 se tratan como empate salvo evidencia más fuerte.
7. Reconocer que el leaderboard público también se ha consultado repetidamente y puede tener sesgo adaptativo. Transferencia al public LB no valida automáticamente generalización al private LB ni a datos clínicos externos.

## Regla de decisión

- Si el stress score no mejora la predicción fuera de familia/tiempo, usarlo como diagnóstico de robustez solamente.
- Si sí mejora de forma consistente, congelar la fórmula y validar un candidato futuro sin volver a modificarla.
- En empates públicos dentro de ±0.003, preferir el modelo con mejor lockbox/OOD solo si su mejora supera la incertidumbre pareada.
- Si score público cae más de 0.003, no asumir que mejor stress lo compensa: conservarlo como candidato de generalización y reportar explícitamente el trade-off.

## Registro

Los resultados deben incluir cohortes evaluables/no evaluables y exclusiones. No reportar correlación sin intervalo, validación por familia y validación temporal.
