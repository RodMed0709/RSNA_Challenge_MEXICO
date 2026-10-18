# exp086_etiquetas_l6_claude

| Campo | Valor |
|---|---|
| Autor | rodrigo (agente labels_claude) |
| Padre | exp054_etiquetas_L5 |
| Fecha | 2026-10-18 |
| Rama | exp086_etiquetas_l6_claude |
| Estado | spec |

Idea (de Rodrigo, autorizada 18-oct): nueva fuente de etiquetas **L6** leyendo los 4407 informes de train con Claude, y
**adjudicación** con un modelo más fuerte de las celdas (estudio × hallazgo) donde L5 (OpenAI) y L6 no concuerdan.
El resultado, **consenso_v6**, es un objetivo de entrenamiento para reentrenar el tanque (final A, exp085). Este
experimento **solo produce etiquetas** (CPU + API, 0 GPU-h, sin notebook de entrenamiento: igual que la fase 1 de
exp054, el generador es `generar_L6.py` en esta carpeta). El uso en entrenamiento lo hace otro experimento y lo decide el LB.

## 1. SPEC (fijada antes de llamar a la API)

### L6

| Punto | Decisión (fija) |
|---|---|
| Modelo | `claude-sonnet-5-5`, k = 1 lectura por informe, sin razonamiento extendido (`thinking: between_tools`, el equivalente a "apagado" en Sonnet 5.5; L5 usó gpt-4.1-mini sin razonamiento). |
| Temperatura | **no se puede fijar**: Sonnet 5.5 y Opus 5.5 rechazan valores no por defecto (400). Desviación declarada frente al brief (pedía 0). |
| Prompt | **B_reason de L5, verbatim** (`exp054/generar_L5.py`, en inglés): mejor lectura individual de L5 contra el consenso de 4 (macro AUC 0.986 vs A 0.969, C 0.983). Notas ≤ 60 palabras + 12 probabilidades enteras 0–100 con la misma guía de calibración y las definiciones oficiales (discusión 733343). Salida JSON con esquema estricto (`output_config.format`). |
| Esquema | 12 hallazgos, mismos nombres y orden que L5; probabilidad = entero / 100. Mismo recorte del informe (12 000 caracteres). |
| API | Message Batches (50 %). Si un batch tarda > 2 h: se cancela y lo que falte va por llamadas síncronas concurrentes (8 hilos). Header `anthropic-workspace-id` en cada request. |

### Acuerdo y adjudicación (umbrales fijos)

- Celda en **acuerdo** si L5 y L6 están del mismo lado de 0.5 (positivo = p > 0.5) **y** |L5 − L6| ≤ 0.30.
  Objetivo afilado: **0.95** si positivo, **0.05** si negativo.
- Si no ⇒ **desacuerdo** ⇒ adjudicación con `claude-opus-5-5` (`effort: medium`, razonamiento adaptativo, que en
  Opus 5.5 no se puede apagar). Una llamada por estudio con todas sus celdas en desacuerdo. Recibe el informe y las dos
  lecturas como "Reading 1 / Reading 2" en **orden aleatorio por estudio** (semilla 86 + UID), sin decir qué modelo es
  cuál. Instrucción adversarial: argumentar contra cada lectura (≤ 40 palabras cada una), decidir una probabilidad
  0–100 con la misma guía de calibración, y responder `ambiguous` si el informe es genuinamente ambiguo.
  - `decided` ⇒ objetivo = probabilidad de Opus / 100.
  - `ambiguous` ⇒ objetivo = media(L5, L6).
  - Adjudicación ausente (error, rechazo, fuera de presupuesto) ⇒ media(L5, L6) (se cuenta y se reporta).
- Estudio sin L6 (error persistente) ⇒ objetivo = L5 de esa celda (se cuenta).
- **Gold (58)**: etiqueta gold dura 0/1 (decisión de Rodrigo, como exp085). L6 y la adjudicación sí se calculan en los
  58 para el reporte, pero gold no ajusta nada (ni prompts, ni umbrales, ni mapeos).
- Salida: `/workspace/users/rodrigo/orq/labels_claude/consenso_v6.csv` (StudyInstanceUID + 12 columnas, probabilidad
  en [0, 1], 4407 filas: mismo formato que las fuentes que lee el notebook de exp044/exp085) y `READY.json`
  {path, n_studies, n_adjudicated, cost_usd, created_at}. L6 suelta: `.../labels_claude/work/L6_sonnet55.csv`. Datos:
  no van a git (licencia MIRA).

### Presupuesto y corte

- Tope **US$40** de API en total. Piloto síncrono de 50 informes (muestra semilla 86) + adjudicación de sus desacuerdos;
  se proyecta a 4407 con precio de batch (Sonnet 5.5 US$2/10 por M, Opus 5.5 US$4/20; batch −50 %).
- Si la proyección total > US$40: adjudicador a `effort: low` y guardia de presupuesto (se adjudican primero los estudios
  con mayor |L5 − L6|; los que no quepan ⇒ media(L5, L6)). Si L6 sola no cabe ⇒ parar (`OVER_BUDGET`) sin L6.
- Fecha dura: etiquetas en el volumen antes del 19-oct 05:00 UTC; si no se llega, se entrega lo que haya y se dice.

### Validación (sin gold para decidir; criterios fijos antes de la corrida)

- L6 **sana** si: cobertura ≥ 99 % de los 4407 con las 12 celdas; macro AUC de L6 contra el consenso de 4 de exp031
  (Raptor+L2+L3+L4, media por celda > 0.5) ≥ 0.93 en los no-gold; ningún hallazgo < 0.80; costo ≤ US$40.
  (Mismos chequeos que L5 en exp054.)
- `READY.json` solo se escribe si L6 está sana. Si no: `didnt`, sin READY.
- **Informativo (no selecciona nada):** macro y por hallazgo AUC contra gold-58 de L5, L6, consenso actual (4 fuentes,
  el de exp085; Raptor no cubre gold), consenso de 5 (con L5) y consenso_v6 (antes de sustituir gold); % de acuerdo
  global y por hallazgo; nº de celdas/estudios adjudicados, `ambiguous`, ausentes; media del objetivo; costo real.
  Si consenso_v6 queda > 0.02 por debajo del consenso actual en gold, se anota como `warning` en READY.json
  (no veta: el uso lo decide otro experimento y el LB).
- **Verdict de este experimento:** `worked` = L6 sana y READY a tiempo; `didnt` = L6 no sana o no llegó a tiempo.

**Sesgos y riesgos declarados:**
- Afilar a 0.95/0.05 las celdas en acuerdo **cambia la escala** del objetivo frente al consenso blando de 4 fuentes
  (más confiado). Es lo que pide la idea; el entrenamiento lo juzgará el LB.
- L5 y L6 usan las **mismas definiciones estrictas** y el mismo prompt B: su acuerdo no es independiente del prompt.
- Solo dos lectores: L2/L3/L4/Raptor no entran en consenso_v6.
- Informes en 10 idiomas; el adjudicador ve los números de ambas lecturas (anclaje posible).
- Sin temperatura 0: la lectura no es exactamente reproducible.
- Afilar a 0.95/0.05 crea **empates masivos**: la AUC de consenso_v6 contra gold (informativa) baja por los empates
  aunque las etiquetas sean igual de correctas (prueba en seco del build con L6 falsa = L5 B_reason y sin adjudicar:
  macro gold 0.840 vs 0.895 del consenso de 4). Por eso esa AUC no puede decidir nada aquí; la medida útil es el LB.

**Presupuesto:** ≤ US$40 de API, 0 GPU-h. Corre en el pod CPU rsna-cron (clave en `/root`, fuera del volumen).

## 2. PLAN

- `generar_L6.py all --out /workspace/users/rodrigo/orq/labels_claude/work`: piloto (50, síncrono) → L6 (batch) →
  adjudicación (batch) → build (CSV, `metrics.json`, READY.json). Cachés jsonl reanudables; log en
  `orq/labels_claude/status.log`.
- `results/exp086_etiquetas_l6_claude/metrics.json` = copia del `metrics.json` del build (solo números agregados).
- Nada más cambia (no hay notebook de entrenamiento en este experimento).

## 3. VERDICT (números de `results/exp086_etiquetas_l6_claude/metrics.json`)

(pendiente)
