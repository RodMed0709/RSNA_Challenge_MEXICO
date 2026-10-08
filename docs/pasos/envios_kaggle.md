# Envíos a Kaggle

Registro de cada envío: qué modelo, LB público, puesto y runtime. Los kernels y logs viven fuera del repo (scratchpad de la
sesión); aquí solo los números. Kernels privados, sin internet, GPU T4.

| Fecha | ID envío | Kernel | Modelo | LB público | Puesto aprox. | Runtime notebook (test visible, 3 estudios) |
|---|---|---|---|---|---|---|
| 2026-10-08 | 56963705 | rsna-knee-exp002-inference v2 | exp002, ensamble 5 folds (8 épocas) | 0.812 | ~4368 / 5530 | 47.2 s |
| 2026-10-08 | 56969845 | rsna-knee-exp015-inference v1 | exp015, solo fold 0 (16 épocas) | 0.810 | ~4374 / 5530 | 45.0 s |

## Lectura

- 1 fold de 16 épocas empata con el ensamble de 5 folds de 8 épocas, con 1/5 del cómputo del modelo (forward 0.52 s/estudio
  contra 0.94 s). Para el track de eficiencia, el ensamble no paga.
- La brecha con el LB es grande: mediana del LB 0.936, nosotros 0.81, mientras la pseudo AUC interna sube a 0.85. Lo que
  más mueve la validación interna (entrada, épocas) apenas toca el LB. Hipótesis principal: las etiquetas de Kaggle siguen
  definiciones estrictas (ACL y MCL solo alto grado o completa; derrame y Baker solo moderado o grande; OA con pérdida de
  cartílago >50 %) y nuestras etiquetas v1 no. La validación interna mide contra v1, no contra la definición del test.
  Esto sube la prioridad de exp011 (etiquetas v2 por LLM con las definiciones oficiales) por encima de más arquitectura.
- En el notebook, leer DICOM domina (22 s de workers en 3 estudios); el forward es menor. Coincide con
  `costo_inferencia.md`: el preprocesado manda el runtime.
