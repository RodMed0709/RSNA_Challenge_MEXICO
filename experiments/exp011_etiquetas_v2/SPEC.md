# exp011_etiquetas_v2

| Campo | Valor |
|---|---|
| Autor | rodrigo |
| Padre | exp001_preparacion |
| Fecha | 2026-10-08 |
| Rama | exp011_etiquetas_v2 |
| Estado | spec |

Idea: comparar etiquetadores contra el dev humano de exp010 y producir `report_labels_v2.csv`.

Contexto en `docs/pasos/paso4_base_solida.md`. Depende de exp010 (el dev). La parte que no depende del dev
(bajar públicos, correr GPT sobre 4,407 informes) puede adelantarse.

## 1. SPEC (antes de tocar el notebook)

**Hipótesis:** las etiquetas v1 (Qwen2.5-7B, "no mencionado" = 0.5) son el techo del modelo. Un etiquetador mejor,
medido contra lectura humana del informe, sube ese techo.

**Candidatos (fijos):**

| id | Etiquetador | Costo |
|---|---|---|
| L0 | v1 tal cual (`report_labels_v1.csv`) | 0 |
| L1 | v1 con "no mencionado" = 0 (incierto sigue 0.5) | 0 |
| L2 | `stevenleehans/rsna-knee-llm-report-labels` | 0 |
| L3 | `pilkwang/rsna-knee-llm-labels` | 0 |
| L4 | `lixin73/rsna-knee-llm-report-labels-sol56` (GPT-5.6-Sol) | 0 |
| L5 | `dreaddevelopment/rsna-knee-labels` (blandas) | 0 |
| L6 | `laymond/rsna-knee-abnormality-qwen3-8b-weak-labels` (Qwen2.5-32B) | 0 |
| L7 | GPT por API, prompt nuevo (abajo) | ≤ US$40 |
| L8 | Consenso: promedio de los 3 mejores de L0–L7 en el dev | 0 |

Prompt L7: definición oficial de cada etiqueta (mismo glosario de exp010), estado `present/absent/uncertain/not_mentioned`,
lado de la rodilla, frase textual de evidencia por etiqueta, salida JSON validada. Temperatura 0. Una llamada por informe.
Conversión: present 1, absent 0, not_mentioned 0, uncertain 0.5. Si la frase de evidencia no aparece en el informe,
la etiqueta se marca como alucinación, se cuenta y queda en 0.5 (incierto); no se borra, para no perder cobertura.

**Métrica de selección:** macro AUC sobre las 12 etiquetas contra el dev de exp010, excluyendo `uncertain` del dev.
Etiquetas con < 8 positivos en el dev se reportan aparte y no entran en la macro. Desempate (Δ < 0.005): menor costo.

**Criterios (no cambian después de correr):**
- Gana el candidato con mayor macro AUC en el dev. Se escribe como `report_labels_v2.csv` (formato de v1).
- Veto gold: si el ganador queda > 0.02 por debajo de L0 en macro AUC contra los 58 gold, se descarta y pasa el siguiente.
  Gold no ordena candidatos, solo veta.
- `worked` si el ganador supera a L0 en el dev por ≥ +0.02. Si no, `didnt` y v1 sigue.
- Cobertura: un candidato que no cubre ≥ 99 % de los 4,407 estudios queda fuera.

**Presupuesto:** 0 GPU-h · API ≤ US$40.

**Riesgos:**
- Algunos públicos pueden haber usado gold para afinar: se revisa su descripción y se anota.
- Mandar informes a la API está permitido (regla 2.6.b, respuesta del host). Nada de los informes va a git.
- Los públicos se declaran en el paper con su URL.

## 2. PLAN

- Celdas: 0 parámetros · 1 setup · 2 cargar y armonizar candidatos al formato de v1 (12 columnas, 0/0.5/1 o prob) ·
  3 GPT sobre 4,407 informes (reanudable, cache por estudio) · 4 evaluar contra dev y gold · 5 elegir y escribir v2 ·
  6 `metrics.json` · 7 verdict.
- Smoke con `SMOKE = True` (20 informes en GPT, dev parcial).

## 3. VERDICT (números de `results/exp011_etiquetas_v2/metrics.json`)

**Resultado:** worked / didnt / killed

| Candidato | Macro AUC dev | Macro AUC gold (veto) | Cobertura | Alucinaciones |
|---|---|---|---|---|
| | | | | |

**Qué aprendimos (2-4 líneas):**

**Siguiente paso sugerido:** exp013 (receta exp002 con etiquetas v2).
