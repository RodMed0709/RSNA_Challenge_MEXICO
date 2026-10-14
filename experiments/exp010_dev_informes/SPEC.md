# exp010_dev_informes

| Campo | Valor |
|---|---|
| Autor | rodrigo |
| Padre | exp001_preparacion |
| Fecha | 2026-10-06 |
| Rama | exp010_dev_informes |
| Estado | spec |

Idea: dev de 200 informes etiquetados por humanos desde el texto, para elegir etiquetador (exp011) sin usar gold.

Contexto en `docs/pasos/paso4_base_solida.md`.

## 1. SPEC (antes de tocar el notebook)

**Problema:** para comparar etiquetadores (Qwen v1, datasets públicos, GPT) no sirve la pseudo AUC, porque cada
uno se mide contra sus propias etiquetas. Elegir con gold rompe la regla 6. Hace falta una referencia propia,
independiente de cualquier etiquetador y fuera de gold.

**Qué produce:** `dev_informes_v1`, 200 estudios × 12 etiquetas leídos por personas desde el informe (no la imagen).
Es un experimento de preparación: no entrena modelo. No hay padre con AUC que comparar.

**Diseño (fijo):**

| Punto | Decisión |
|---|---|
| Muestra | 200 estudios del fold 0 de `split_v1_seed42`, nunca fold −1. Semilla 42. |
| Estratificación | por idioma del informe (proporcional, mínimo 10 por idioma si existen) y enriquecida para que cada etiqueta tenga ≥ 8 positivos según la unión de etiquetadores disponibles (v1 + públicos). La unión solo elige qué estudios entran, no decide ninguna etiqueta. |
| Traducción | GPT traduce cada informe al inglés. **GPT no etiqueta en este experimento**: si las personas vieran etiquetas de GPT, el dev quedaría sesgado a favor de GPT en exp011. |
| Anotación | Alejandra y José Javier leen el original + la traducción y marcan por etiqueta: `present`, `absent`, `not_mentioned`, `uncertain`, más la frase que lo justifica. Glosario con las definiciones oficiales de las 12 etiquetas. |
| Reparto | 120 estudios cada uno; 40 en común para medir acuerdo entre anotadores. Rodrigo resuelve los desacuerdos de los 40. |
| Conversión | `present` = 1; `absent` y `not_mentioned` = 0; `uncertain` se excluye al calcular AUC en exp011. |
| Dónde vive | `/workspace/data/labels/dev_informes_v1.csv` (dato, no va a git). En git solo `metrics.json`. |

**Criterios (no cambian después de correr):**
- El kappa que decide es el binario (`present` contra el resto), promedio de las 12 etiquetas en los 40 comunes:
  es la forma en que el dev se usa en exp011. El kappa de 4 categorías se reporta, no decide.
- Éxito (`worked`): 200 estudios completos; kappa de Cohen macro en los 40 comunes ≥ 0.80; cada etiqueta con ≥ 8 positivos.
- Parcial: kappa entre 0.60 y 0.80 → se usa, pero se revisa el glosario y se reportan las etiquetas débiles.
- Kill: kappa < 0.60 → el glosario no sirve; se reescribe y se repite (nuevo experimento).
- Etiqueta con < 8 positivos: se reporta y en exp011 se evalúa aparte (no entra en la macro).

**Presupuesto:** 0 GPU-h · API ≤ US$5 (200 traducciones) · ~8 h-persona de anotación.

**Riesgos:**
- Sesgo de anclaje: mitigado porque GPT solo traduce.
- La traducción puede suavizar matices (p. ej. "no se descarta"): el anotador ve también el original.
- Informes y traducciones son datos de la competencia: nada de esto se sube a git ni fuera del equipo.
- El dev mide extracción del informe, no verdad de imagen. La distancia informe vs imagen la sigue vigilando gold.

## 2. PLAN

- Notebook `exp010_dev_informes.ipynb`, celdas: 0 parámetros · 1 setup · 2 muestreo estratificado ·
  3 traducción GPT (reanudable, cache por estudio) · 4 exportar hoja de anotación (`.xlsx` con listas desplegables)
  y glosario · 5 importar anotaciones, validar y calcular kappa · 6 guardar `metrics.json` · 7 verdict.
- Modelo de traducción y costo real quedan en `metrics.json`. La key va en `.secrets.env` como `OPENAI_API_KEY`.
- Smoke con `SMOKE = True` (5 informes) antes de traducir los 200.

## 3. VERDICT (números de `results/exp010_dev_informes/metrics.json`)

**Resultado:** worked / didnt / killed

| Métrica | Valor |
|---|---|
| Estudios anotados | |
| Kappa macro (40 comunes) | |
| Etiquetas con < 8 positivos | |
| Costo API | |
| Horas-persona | |

**Qué aprendimos (2-4 líneas):**

**Siguiente paso sugerido:** exp011_etiquetas_v2.
