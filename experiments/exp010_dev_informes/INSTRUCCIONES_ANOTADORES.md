# Instrucciones para anotar los informes de exp010

Cada fila corresponde al informe de **una sola rodilla**. Lee primero el informe original y usa la traducción al inglés como apoyo. Si notas que la traducción cambia el sentido, confía en el original y explícalo en `evidencia / notas`.

Para cada una de las 12 etiquetas elige exactamente un estado:

- `present`: el informe afirma que el hallazgo está presente y cumple la definición del glosario.
- `absent`: el informe lo niega de forma explícita. Ejemplo: “no fracture” → `Fracture = absent`.
- `not_mentioned`: el informe no habla de ese hallazgo. No supongas que está sano.
- `uncertain`: el informe lo presenta como posible o dudoso. Ejemplos: “possible tear”, “cannot be excluded”, “suggestive of”.

## Cómo trabajar

1. Abre tu archivo `exp010_anotacion_<nombre>.xlsx` y usa solo la hoja `anotar`.
2. Lee la definición de cada etiqueta en la hoja `glosario`.
3. Completa las 12 celdas con el menú desplegable. No dejes ninguna vacía.
4. En `evidencia / notas`, copia la frase más útil del informe o anota la duda de traducción. No hace falta escribir una nota distinta por etiqueta.
5. Guarda el mismo archivo y devuélvelo a Rodrigo. No cambies nombres de hojas, columnas ni estudios.

## Ejemplos rápidos

- “Complete ACL rupture” → `ACL = present`.
- “ACL intact” → `ACL = absent`.
- El informe habla de meniscos y ligamentos, pero nunca de fractura → `Fracture = not_mentioned`.
- “A subtle meniscal tear cannot be excluded” → el menisco correspondiente = `uncertain`.
- “Mild MCL sprain” no cumple la definición oficial de lesión MCL de alto grado → `MCL = absent`; explica “esguince leve” en notas.
- “Small joint effusion” no cumple la definición oficial de derrame moderado o grande → `Effusion = absent`.

Izquierda o derecha no cambia la decisión: cada fila ya representa un estudio de una sola rodilla. Lo importante es la estructura o compartimento (`medial`, `lateral` o `patelofemoral`) que menciona el informe.

Si una frase no encaja claramente en estas reglas, usa `uncertain` y deja una nota para que Rodrigo la revise.
