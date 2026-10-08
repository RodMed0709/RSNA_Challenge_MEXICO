# Bitácora — exp011_etiquetas_v2

Estado al 2026-10-08: **notebook listo, sin corrida real**. L0–L6 se descargan y armonizan; L7 (GPT) no ha hecho
ninguna llamada real porque falta la `OPENAI_API_KEY`; la evaluación espera el dev humano de exp010.
Ningún AUC de este archivo existe todavía. Los resultados se copian de `metrics.json` cuando existan.

## 1. Qué es y para qué sirve

Las etiquetas que hoy entrenan todos los modelos (`report_labels_v1.csv`, Qwen2.5-7B, exp001) salen de leer el informe
con un LLM. Si esa lectura tiene errores, el modelo aprende esos errores: las etiquetas son el techo. Hay varias
alternativas gratis (datasets públicos de Kaggle) y una de pago (GPT con prompt nuevo). Este experimento las compara
**todas contra la misma hoja de respuestas** y se queda con la mejor como `report_labels_v2.csv`.

**Por qué decide el dev y no gold.** La regla 6 prohíbe que gold (58 estudios, fold −1) elija nada: solo veta.
La pseudo AUC tampoco sirve, porque cada etiquetador se mediría contra sus propias etiquetas. El dev de exp010
(200 informes del fold 0 anotados por Alejandra y José Javier con las definiciones oficiales) es la referencia
neutral y propia. Gold solo descarta a un ganador que quede > 0.02 por debajo de L0.

El experimento no entrena modelo ni usa GPU. Mide qué tan bien se **lee el informe**, no la verdad de la imagen.

## 2. Registro cronológico

| Fecha | Qué pasó | Commit |
|---|---|---|
| 2026-10-08 | SPEC escrita (secciones 1 y 2): candidatos L0–L8 fijos, métrica de selección, criterios, presupuesto ≤ US$40. | `5b0fcbc` |
| 2026-10-08 | Notebook construido por el worker (Codex), 8 celdas. Verificación declarada por el worker: 5 descargas Kaggle OK; corrida sin dev exit 0; corrida con un dev falso de 200 estudios exit 0 y salida (4407, 14), dev falso borrado después; contrato de cache/presupuesto de L7 PASS; pytest 8 passed / 1 skipped; sin dependencias nuevas. **Ninguna llamada real a OpenAI** (no hay key). | `7cf1609` |
| 2026-10-08 | Revisión del orquestador: tres notas (abajo) y la tabla de cobertura. | — |

### Notas de la revisión del orquestador (2026-10-08)

- **(a) L1 es una aproximación.** v1 guarda `0.5` tanto para "incierto" como para "no mencionado" y no hay campo que
  los distinga. Por eso L1 convierte **todo** `0.5` a `0`, no solo el "no mencionado" que pedía la SPEC. Está
  documentado en la celda 2 del notebook. Es una desviación forzada de la SPEC y se declara en el verdict.
- **(b) L2 y L6 tocaron gold.** Las fichas de `stevenleehans` (L2) y `laymond` (L6) declaran que ajustaron prompt o
  reglas mirando los 58 gold. Su número de veto gold está **inflado**. No afecta la selección (decide el dev), pero se
  declara en el verdict y en el paper.
- **(c) Modelos y definiciones.** L7 usa las definiciones oficiales estrictas (discusión 733343), las mismas del
  glosario de exp010. L3 usa `claude-opus-5` según su código generador.

## 3. Candidatos

Cobertura = estudios de `train.csv` (4,407) con las 12 etiquetas completas. Regla de la SPEC: < 99 % → fuera.

| id | Origen | Modelo | Riesgo de gold | Cobertura |
|---|---|---|---|---|
| L0 | `report_labels_v1.csv` (exp001) | Qwen2.5-7B-Instruct | no (gold solo se usó para evaluar después) | v1 completo* |
| L1 | v1 con todo `0.5` → `0` | Qwen2.5-7B-Instruct | no | igual que L0* |
| L2 | `stevenleehans/rsna-knee-llm-report-labels`, `llm_labels_v4_blend.csv` | no identificado (blend v2 + GPT-5.6-Sol) | **sí, declarado** (validación y ajustes de Synovitis sobre gold) | 4,407 (100 %) |
| L3 | `pilkwang/rsna-knee-llm-labels`, `report_labels_v2.csv` | `claude-opus-5` (según su código) | no indicado | 4,406 (99.98 %) |
| L4 | `lixin73/rsna-knee-llm-report-labels-sol56`, `labels_llm_gpt56sol.csv` | GPT-5.6-Sol | no indicado (ficha vacía) | 4,407 (100 %) |
| L5 | `dreaddevelopment/rsna-knee-labels`, `labels_llm_soft.csv` | LLM no especificado, probabilidades blandas | no indicado | 4,349 (98.68 %) → **excluido** |
| L6 | `laymond/rsna-knee-abnormality-qwen3-8b-weak-labels`, `weak_labels_v2_filled.csv` (`__confidence`) | Qwen2.5-32B-Instruct 4-bit | **sí, declarado** (prompt y reglas auditados contra gold) | 4,407 (100 %) |
| L7 | GPT por API, prompt nuevo | `gpt-5-mini` (por defecto) | no | pendiente (sin key) |
| L8 | Promedio de los 3 mejores en el dev con cobertura ≥ 99 % | — | hereda el de sus miembros | pendiente (necesita dev) |

\* La revisión no reportó el número de cobertura de L0/L1; sale en `metrics.json` (`coverage`) en la primera corrida.

## 4. Cómo correrlo y en qué orden respecto a exp010

### Archivos (todos fuera de git, bajo el data root)

Data root: `RSNA_DATA` (por defecto `/workspace/data`), o `DATA_ROOT_OVERRIDE` en la celda 0.

| Archivo | Ruta | Lo crea |
|---|---|---|
| Entradas | `raw/train.csv`, `labels/report_labels_v1.csv` | ya existen |
| Datasets públicos | `labels/public/<usuario>/<dataset>/<archivo>.csv` | celda 2 (descarga si no está) |
| Cache GPT | `labels/exp011_gpt_cache/<StudyInstanceUID>.json` | celda 3 |
| Dev de exp010 | `labels/dev_informes_v1.csv` | exp010, celda 5 |
| Etiquetas v2 | `labels/report_labels_v2.csv` (4,407 filas, 12 etiquetas + `source`) | celda 5, solo con dev |
| Métricas (sí va a git) | `results/exp011_etiquetas_v2/metrics.json` (o `RSNA_RESULTS`) | celda 6 |

Secretos: `OPENAI_API_KEY` en `.secrets.env` en la raíz del repo (en `.gitignore`); `KAGGLE_API_TOKEN` en el entorno
si los públicos no están descargados. El notebook nunca los imprime.

### Qué se puede correr ya (sin dev)

1. **Smoke:** celda 0 con `SMOKE = True`, correr celdas 0–6. Sin dev, L7 toma 20 informes al azar de los 4,407.
   Revisar a mano en el cache que las frases de evidencia sean textuales y que los estados respeten la definición
   estricta (p. ej. "derrame pequeño" → `absent`).
2. **L7 completo:** en cuanto haya key, `SMOKE = False` y correr celdas 0–6, o sin navegador
   `bash run_notebook.sh experiments/exp011_etiquetas_v2/exp011_etiquetas_v2.ipynb`. Procesa los 4,407 informes con
   8 llamadas en paralelo y hasta 5 reintentos. Es reanudable: lo que está en cache con el mismo modelo y prompt no se
   vuelve a pedir. Se detiene solo antes de pasar de US$40.
3. Sin dev, las celdas 4–5 imprimen `dev pendiente`: no se elige ganador ni se escribe `report_labels_v2.csv`.
   `metrics.json` queda con `decision_auto = "pending_dev"`. Eso es una corrida preparatoria, no un verdict.

### Qué espera al dev

4. Cuando exista `labels/dev_informes_v1.csv` (exp010 cerrado), volver a correr celdas 0–6 con `SMOKE = False`.
   L7 sale del cache (costo extra cero). Se calculan las AUC dev y gold, se arma L8, se elige ganador y se escribe v2.
5. Verdict: copiar números de `metrics.json` a la sección 3 de `SPEC.md` y a `docs/EXPERIMENTOS.md`.

## 5. Decisiones y por qué

- **Decide el dev, gold solo veta** (regla 6). Gold no ordena candidatos.
- **Macro AUC sin `uncertain` del dev; etiquetas con < 8 positivos fuera de la macro** (se reportan aparte): con pocos
  positivos la AUC es ruido.
- **Empate (Δ < 0.005) → el más barato.** Un público gratis gana a GPT si empatan.
- **`worked` solo si el ganador supera a L0 por ≥ +0.02.** Si no, `didnt` y `report_labels_v2.csv` se escribe con L0
  (v1 sigue).
- **Cobertura ≥ 99 %:** un etiquetador al que le faltan estudios obligaría a mezclar fuentes.
- **L7 con salida JSON estricta (Structured Outputs) y cita textual por etiqueta.** Si la cita no aparece en el
  informe, la etiqueta se cuenta como alucinación y se deja vacía (NaN), no se adivina.
- **L7 con definiciones oficiales estrictas**, iguales al glosario del dev, para que GPT y los anotadores midan lo mismo.
- **`gpt-5-mini` sin `temperature`:** la SPEC pedía temperatura 0, pero los GPT-5 originales rechazan ese parámetro.
  El notebook solo lo envía si el modelo lo acepta y registra `openai_temperature_sent` en `metrics.json`.
- **L1 convierte todo 0.5 a 0** (ver nota a): es lo único posible con el formato de v1.
- **L6 usa `__confidence` como P(present)**, la única columna continua comparable del dataset.

## 6. Pendientes y riesgos

- **Falta la `OPENAI_API_KEY`.** L7 no tiene ninguna llamada real; costo, alucinaciones y cobertura de L7 son
  desconocidos.
- **Costo de L7.** El tope de US$40 está en código: antes de cada llamada se reserva el peor caso (payload completo
  + 4,500 tokens de salida) y se aborta si la suma pasaría el tope. Las tarifas de `gpt-5-mini` están fijas en la
  celda 0 (consultadas el 2026-10-08); si OpenAI las cambia, el control de costo queda mal calibrado. El costo real
  sale de `metrics.json` (`l7.usage.cost_usd`).
- **Alucinaciones bajan la cobertura de L7.** Una etiqueta alucinada queda NaN y el estudio deja de contar como
  completo; con suficientes alucinaciones L7 puede caer bajo 99 % y quedar fuera.
- **Definiciones estrictas vs informes que no gradúan.** Muchos informes dicen "derrame" o "rotura" sin decir tamaño
  ni grado. Con la definición estricta eso puede quedar como `absent`, `uncertain` o `present` según cómo lo lea cada
  etiquetador o anotador. Es fuente de desacuerdo con el dev y entre candidatos; mirar el detalle por etiqueta.
- **L5 excluido** por cobertura (4,349, 98.68 % < 99 %). Se reporta pero no puede ganar ni entrar a L8.
- **L2 y L6 con gold inflado** (nota b). Su número gold no prueba nada; declararlo en verdict y paper.
- **L1 no es exactamente lo que dice la SPEC** (nota a). Declararlo en el verdict.
- **El dev mide lectura del informe, no imagen.** Un etiquetador puede ganar el dev y no mejorar el modelo; eso se
  comprueba en exp013 (receta exp002 con etiquetas v2).
- **Datos sensibles:** caches GPT, públicos y `report_labels_v2.csv` nunca van a git. Los públicos se citan en el
  paper con su URL.

## 7. Resultados (de `results/exp011_etiquetas_v2/metrics.json`)

| Candidato | Macro AUC dev | Macro AUC gold (veto) | Cobertura | Costo (USD) | Alucinaciones |
|---|---|---|---|---|---|
| L0 | | | | | |
| L1 | | | | | |
| L2 | | | | | |
| L3 | | | | | |
| L4 | | | | | |
| L5 | | | | | |
| L6 | | | | | |
| L7 | | | | | |
| L8 | | | | | |

| Campo | Valor |
|---|---|
| Miembros de L8 | |
| Ganador tras veto gold | |
| Δ dev vs L0 | |
| Etiquetas fuera de la macro (< 8 positivos) | |
| Fuente de `report_labels_v2.csv` | |
| Resultado (`worked` / `didnt` / `killed`) | |
