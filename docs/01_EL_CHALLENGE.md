# El challenge

Tenemos una resonancia de rodilla con varias series y cortes. Por cada estudio debemos entregar la probabilidad de 12 hallazgos. En test no hay informe radiológico: el modelo final solo puede usar imágenes.

## Las 12 etiquetas

- `ACL`: lesión o rotura del ligamento cruzado anterior, que estabiliza la rodilla hacia adelante.
- `MCL`: lesión del ligamento colateral medial, ubicado en la cara interna.
- `Medial Meniscus`: lesión del menisco interno, el amortiguador del lado medial.
- `Lateral Meniscus`: lesión del menisco externo, el amortiguador del lado lateral.
- `Medial OA`: artrosis en el compartimento interno de la articulación.
- `Lateral OA`: artrosis en el compartimento externo.
- `PF OA`: artrosis patelofemoral, entre rótula y fémur.
- `Effusion`: exceso de líquido dentro de la articulación, o derrame.
- `Synovitis`: inflamación de la membrana que recubre la articulación.
- `Baker's`: quiste de Baker, una bolsa de líquido detrás de la rodilla.
- `Contusion`: contusión ósea, parecida a un moretón dentro del hueso.
- `Fracture`: fractura o ruptura del hueso.

## Métrica y entrega

La métrica es el promedio simple de la AUC ROC de las 12 columnas. AUC mide qué tan bien ordenamos positivos arriba de negativos; no hay que escoger un umbral.

La entrega es `submission.csv`: una fila por `StudyInstanceUID`, las 12 columnas en el orden de `sample_submission.csv` y probabilidades entre 0 y 1. Es una competencia de código: el notebook de Kaggle debe terminar en menos de 9 horas, sin internet.

Fechas importantes:

- Unión final del equipo: **15 de octubre de 2026**.
- Entrega final: **22 de octubre de 2026, 23:59 UTC**.

## El problema de las 58 etiquetas

Train tiene 4,407 estudios, pero solo 58 traen las 12 etiquetas de expertos. Entrenar visión únicamente con esos 58 sobreajustaría muy rápido. Los otros estudios sí tienen informe radiológico en texto y esos informes vienen en unos 12 idiomas.

Por eso usamos Qwen para convertir cada informe en 12 pseudo-etiquetas: `1` presente, `0` ausente y `0.5` incierto. No son verdad perfecta; los 58 estudios con etiqueta experta sirven para medir dónde falla esa extracción. Después entrenamos el modelo de imagen con supervisión suave y conservamos NaN donde no hay una etiqueta usable.
