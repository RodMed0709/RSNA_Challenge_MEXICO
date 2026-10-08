# Bitácora — exp010_dev_informes

Estado al 2026-10-08: **notebook listo, sin correr**. Falta la `OPENAI_API_KEY` y la anotación humana.
Ningún número de este archivo sale de una corrida todavía. Los resultados se copian de `metrics.json` cuando existan.

## 1. Qué es y para qué sirve

Es una **hoja de respuestas hecha por personas**: 200 informes del fold 0 donde Alejandra y José Javier marcan, leyendo
el texto, si cada una de las 12 etiquetas está presente o no.

Para qué: en exp011 hay que elegir el mejor etiquetador (Qwen v1, datasets públicos de Kaggle, GPT con prompt nuevo).
No se puede elegir con la pseudo AUC, porque cada etiquetador se mide contra sus propias etiquetas. Tampoco con gold,
porque la regla 6 lo prohíbe: gold (58 estudios, fold −1) **nunca selecciona nada**, solo veta. Este dev es la
referencia propia y neutral que permite comparar etiquetadores sin tocar gold.

No entrena modelo, no usa GPU. El dev mide qué tan bien se **lee el informe**, no la verdad de la imagen.

## 2. Registro cronológico

| Fecha | Qué pasó | Commit |
|---|---|---|
| 2026-10-06 | SPEC escrita (secciones 1 y 2) junto con el plan del paso 4. | `74b1c48` |
| 2026-10-08 | Notebook construido por el worker (Codex): muestreo, traducción, libros XLSX, importación, kappa, adjudicación, CSV final. Se añade `INSTRUCCIONES_ANOTADORES.md` y `openai` + `openpyxl` a `requirements.txt`. | `48b2ebd` |
| 2026-10-08 | Revisión del orquestador. Verificación declarada por el worker: pytest 8 passed / 1 skipped; integración simulada sin API PASS (2 XLSX de 120, 40 en común, hoja de adjudicación, CSV de 200); las 7 celdas de código compilan; outputs vacíos. **No hubo llamada real a OpenAI** (no hay key). | — |
| 2026-10-08 | Hallazgo: las definiciones oficiales de Kaggle son estrictas (ver abajo). | — |

### Hallazgo 2026-10-08: las definiciones oficiales son estrictas

Fuente: discusión oficial de la competencia, https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733343
(copiadas al glosario de los XLSX, celda 4 del notebook).

- **ACL y MCL:** solo rotura de alto grado o completa. Esguince leve, cambio de señal o engrosamiento = negativo.
- **Derrame (Effusion) y quiste de Baker:** solo moderado o grande. "Derrame pequeño" = negativo.
- **Artrosis (Medial/Lateral/PF OA):** solo pérdida de cartílago de alto grado (> 50 % del grosor) en un área de ~1 cm o más.

Consecuencia probable: las etiquetas v1 (Qwen2.5-7B, exp001) marcaron como positivos hallazgos leves que la definición
oficial cuenta como negativos. Eso puede explicar parte del ~0.80 de acuerdo de v1 con gold. Es una **hipótesis**, no
un resultado; se comprueba en exp011, donde es hipótesis central. Para este dev, los anotadores siguen la definición
estricta (las instrucciones ya incluyen "esguince leve MCL → absent" y "derrame pequeño → absent").

## 3. Flujo operativo paso a paso

### Archivos (todos fuera de git, bajo el data root)

El data root es `DATA_ROOT` de `src/knee/config.py`: variable de entorno `RSNA_DATA`, por defecto `/workspace/data`.

| Archivo | Ruta | Lo crea |
|---|---|---|
| Entradas | `raw/train.csv`, `cache/meta_v2/study_meta_v2.csv` (idioma, de exp009), `labels/report_labels_v1.csv` | ya existen |
| Muestra de 200 | `labels/exp010_muestra.csv` | celda 2 |
| Cache de traducciones | `labels/exp010_cache/<StudyInstanceUID>.json` (un JSON por estudio) | celda 3 |
| Libros de anotación | `labels/exp010_anotacion/exp010_anotacion_alejandra.xlsx`, `..._josejavier.xlsx` | celda 4 |
| Hoja de adjudicación | `labels/exp010_anotacion/exp010_adjudicacion.xlsx` | celda 5 |
| Dev final | `labels/dev_informes_v1.csv` | celda 5 |
| Métricas (sí va a git) | `results/exp010_dev_informes/metrics.json` (en el repo) | celda 6 |

La key va en `.secrets.env` en la raíz del repo (`OPENAI_API_KEY=...`). Ese archivo está en `.gitignore`.

### Paso A — Rodrigo prepara (celdas 0–4)

1. En el pod o en local. En local hay que apuntar `RSNA_DATA` a una copia del data root con las tres entradas.
2. Poner `OPENAI_API_KEY` en `.secrets.env`.
3. Smoke: celda 0 con `SMOKE = True` (traduce solo 5) y correr celdas 0–4. Revisar a mano que las 5 traducciones
   conservan negaciones y dudas.
4. Corrida real: `SMOKE = False` y volver a correr 0–4. La traducción es reanudable: lo que ya está en cache no se
   vuelve a pedir. Si los XLSX ya existen, no se sobrescriben; solo se rellenan las traducciones vacías.
   Opción sin navegador: `bash run_notebook.sh experiments/exp010_dev_informes/exp010_dev_informes.ipynb`.
5. Comprobar que las 200 traducciones están (la celda 3 imprime `Traducciones disponibles: N/200`) y que la celda 2
   no falló (exige ≥ 8 positivos por etiqueta en la unión y mínimo de idiomas).
6. Enviar a cada anotador **su** XLSX y `INSTRUCCIONES_ANOTADORES.md`. Solo dentro del equipo.

### Paso B — Alejandra y José Javier anotan

- 120 estudios cada uno (80 propios + 40 en común; no saben cuáles son comunes).
- Hoja `anotar`: 12 celdas por fila con menú desplegable (`present`, `absent`, `not_mentioned`, `uncertain`)
  y una nota de evidencia. Hoja `glosario`: definiciones oficiales.
- No cambiar nombres de hojas, columnas ni estudios. Devolver el mismo archivo a Rodrigo.
- Anotar las horas reales dedicadas (hacen falta para el verdict).

### Paso C — Rodrigo consolida (celdas 5–6, dos veces)

1. Copiar los XLSX devueltos a `labels/exp010_anotacion/` con el mismo nombre.
2. Correr 0–2 y 4–6 (la 3 no gasta nada si la cache está completa). La celda 5 valida (estudios, valores, vacíos),
   calcula kappa en los 40 comunes y crea `exp010_adjudicacion.xlsx` con cada desacuerdo.
3. Llenar la columna `adjudicacion` de esa hoja.
4. Volver a correr 5–6. Si todo está adjudicado, se escribe `dev_informes_v1.csv` (200 filas) y `metrics.json` queda
   con `status: complete`.
5. Verdict: copiar números de `metrics.json` a la sección 3 de `SPEC.md`, fila en `docs/EXPERIMENTOS.md`, PR a `main`.

### Tiempos estimados (estimación, no medidos)

| Tarea | Estimado |
|---|---|
| Smoke + corrida de traducción (200 llamadas) | menos de 1 h de Rodrigo |
| Anotación | ~8 h-persona en total según SPEC (~4 h cada uno, 120 informes) |
| Adjudicación de los desacuerdos de 40 estudios | ~1 h |
| Costo API | ≤ US$5 según SPEC; el costo real queda en `metrics.json` |

## 4. Decisiones y por qué

| Decisión | Por qué |
|---|---|
| GPT **solo traduce**, no etiqueta; las instrucciones prohíben resumir o clasificar. | GPT es candidato en exp011. Si los anotadores vieran etiquetas de GPT, se anclarían y el dev favorecería a GPT. |
| El anotador ve original + traducción. | La traducción puede suavizar matices ("no se descarta"); ante duda manda el original. |
| 40 estudios en común. | Permite medir acuerdo entre personas (kappa de Cohen) sin duplicar todo el trabajo. 200 = 40 + 80 + 80. |
| Muestreo estratificado por idioma (`study_meta_v2.lang`), mínimo 10 por idioma. | Que todos los idiomas pesen y el dev no sea solo inglés. |
| Enriquecimiento a ≥ 8 positivos por etiqueta usando la unión de etiquetadores. | Sin positivos no hay AUC por etiqueta. La unión **solo elige qué estudios entran**; la etiqueta la ponen las personas. |
| `not_mentioned` = 0. | Si el informe no lo dice, se trata como ausente (decisión de la SPEC). v1 usaba 0.5 y el modelo aprendía ese ruido. |
| `uncertain` se excluye (NaN). | Un "posible" no es ni 0 ni 1; meterlo en cualquiera de los dos ensucia la AUC de exp011. |
| Fold 0, nunca fold −1. | Gold no puede participar en ninguna selección (regla 6). |
| Glosario con las definiciones oficiales de Kaggle. | El dev tiene que medir lo mismo que mide la competencia. |

## 5. Pendientes y riesgos

- **Falta la key** y por tanto la primera llamada real a OpenAI. El costo por token está fijo en el código solo para
  `gpt-5-mini`; con otro modelo el costo sale `null`.
- **Qué kappa decide.** La celda 5 calcula dos: 4 categorías y "present vs resto". La SPEC dice "kappa macro" sin
  precisar. Hay que fijarlo en la SPEC **antes** de ver resultados (regla 8).
- **No repartir los XLSX del smoke.** La celda 4 los crea ya en la corrida con `SMOKE = True`, con solo 5 traducciones.
  Se rellenan al correr con `SMOKE = False`, pero no deben enviarse antes.
- **Reproducibilidad de la muestra.** La unión usa todos los CSV de `labels/` que haya en disco al correr la celda 2.
  Si cambian esos archivos, cambia la muestra; la celda 4 detecta el cambio y falla en vez de sobrescribir. No borrar
  `exp010_muestra.csv` ni los XLSX una vez enviados.
- **Local vs pod.** En local hay que definir `RSNA_DATA`; si no, apunta a `/workspace/data`.
- **Horas-persona** no se miden en el notebook: los anotadores las reportan a mano.
- **Datos sensibles.** Informes, traducciones y XLSX son datos de la competencia: nada a git ni fuera del equipo.
- El dev mide extracción del informe, no verdad de imagen. La distancia informe–imagen la sigue vigilando gold.

## 6. Resultados (de `results/exp010_dev_informes/metrics.json`)

| Métrica | Valor |
|---|---|
| Estudios anotados | |
| Kappa macro, 4 categorías (40 comunes) | |
| Kappa macro, present vs resto (40 comunes) | |
| Etiquetas con < 8 positivos | |
| Desacuerdos adjudicados | |
| Costo API (USD) | |
| Horas-persona | |
| Resultado (`worked` / `didnt` / `killed`) | |
