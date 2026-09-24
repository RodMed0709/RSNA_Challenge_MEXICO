# Metodología SDD de experimentos

Cada experimento sigue el mismo ciclo. Así cualquiera puede leer qué se probó, por qué y qué pasó, y los resultados
de distintas personas se pueden comparar.

## Ciclo y compuertas

| Fase | Qué produce | Para pasar a la siguiente |
|---|---|---|
| **0. CLAIM** | `nuevo_experimento("tag", padre="expNNN")`: carpeta, notebook copiado del padre, SPEC en blanco, fila `spec` en el registro | Número global reservado |
| **1. SPEC** | Sección 1 de `SPEC.md` | Una sola variable contra el padre, criterios numéricos, presupuesto. Commit de la SPEC **antes** de tocar el notebook |
| **2. PLAN** | Sección 2 de `SPEC.md` | Lista cerrada de celdas que cambian |
| **3. IMPLEMENT** | Notebook | Solo cambian las celdas del PLAN; `SMOKE = True` corre completo |
| **4. RUN** | `results/expNNN_tag/metrics.json` con fold 0 | GPU-h ≤ 1.5 × presupuesto; si no, se para y se escribe el verdict |
| **5. VERDICT** | Sección 3 de `SPEC.md` y fila del registro | Números copiados de `metrics.json`, estado y una línea de aprendizaje |
| **6a. PROMOTE** | 5 folds (OOF) y PR a `main` | Pasa la confirmación; pasa a ser la nueva referencia |
| **6b. ARCHIVE** | PR a `main` con SPEC, notebook y verdict | — (los fracasos también se suben: enseñan) |

## Reglas de decisión (iguales para todos)

La celda 5 del notebook calcula la decisión automática; la persona la confirma en el verdict.

- **Screen (fold 0)** contra el fold 0 del padre:
  - Δ pseudo ≥ +0.005 → `promising`: se corre con 5 folds.
  - −0.005 < Δ < +0.005 → `neutral`: cuenta como `didnt`, salvo que ahorre ≥30 % de cómputo.
  - Δ ≤ −0.005 → `didnt`.
- **Veto gold:** si gold cae más de 0.02 contra el padre → `killed`, diga lo que diga pseudo.
- **Confirmación (5 folds):** la AUC OOF pseudo (5 folds concatenados) supera a la del padre y el veto gold sigue pasando
  (gold con el promedio de los 5 modelos).
- **Gold nunca elige** época, checkpoint ni hiperparámetros. El checkpoint se elige con pseudo del fold de validación.

¿Por qué? Gold son 58 estudios: su AUC promedio se mueve ±0.03–0.04 solo por azar. Pseudo son ~870 estudios por fold,
mucho más estable, aunque mide acuerdo con las etiquetas de Qwen (~0.80 contra gold). Por eso pseudo decide y gold vigila
que no estemos aprendiendo los errores de Qwen.

## Estados del registro

`spec` → `running` → `worked` / `didnt` / `killed`. Arriba de la tabla de `docs/EXPERIMENTOS.md` está la
**referencia actual**; solo cambia con un experimento promovido.

## Celdas fijas del notebook

| # | Contenido |
|---|---|
| 0 | Parámetros (etiqueta `parameters`). Única celda con valores: `EXP`, `PADRE`, `FOLDS`, `SEED`, `SMOKE`, hiperparámetros |
| 1 | Setup: imports, semillas, rutas, git, revisa que exista la SPEC |
| 2 | Datos y split con revisiones: gold fuera, folds válidos |
| 3 | Modelo, optimizador y pérdida |
| 4 | Entrenamiento por fold; guarda `best_f{k}.pt`, `oof_f{k}.csv`, `gold_f{k}.csv`, `history_f{k}.csv` |
| 5 | Evaluación contra el padre y decisión automática |
| 6 | Guarda `metrics.json` |
| 7 | Recordatorio del verdict |

La mayoría de los experimentos cambian la celda 0 (un parámetro) o la 3 (modelo, pérdida). Si el cambio es en datos
(por ejemplo, otra selección de series), va en la celda 2.

## Prompt para Claude sin acceso al repo (claude.ai)

Pega esto al inicio del chat, seguido de la `SPEC.md` de tu experimento y de su padre:

```text
Eres mi asistente para un experimento del equipo RSNA Knee (Kaggle, 12 etiquetas binarias por estudio de
resonancia de rodilla, métrica AUC media). Trabajamos con SDD (spec-driven development). No ves el repo: solo lo
que yo pegue. No inventes rutas, funciones ni números; si te falta algo, pídemelo.

Reglas que DEBES hacer cumplir, aunque yo te pida saltarlas:
1. Fases en orden: SPEC → PLAN → IMPLEMENT → RUN (fold 0) → VERDICT → PROMOTE o ARCHIVE.
   Empieza cada respuesta diciendo en qué fase estamos. No escribas código sin SPEC completa y confirmada.
2. La SPEC tiene: hipótesis, UNA sola variable cambiada contra el padre (valor padre → nuevo), efecto esperado,
   criterios (éxito: Δ pseudo fold 0 ≥ +0.005; kill: Δ ≤ −0.005, o gold < padre − 0.02, o > 1.5× presupuesto),
   presupuesto en horas de GPU y riesgos.
3. Si propongo dos cambios, sepáralos en dos experimentos.
4. El entrenamiento va en celdas del notebook (0 parámetros, 1 setup, 2 datos y split, 3 modelo, 4 entrenamiento,
   5 evaluación, 6 guardar metrics.json, 7 verdict). Solo modifica las celdas del PLAN. No crees archivos .py.
5. El split `split_v1_seed42` no se cambia ni se rehacen folds.
6. Los 58 estudios gold (fold = -1) NUNCA eligen época, checkpoint ni hiperparámetros: solo vetan. SELECT_ON = "pseudo".
7. Los criterios de la SPEC no se cambian después de ver resultados.
8. El VERDICT usa solo números que yo pegue de metrics.json: tabla padre / este / Δ y 2-4 líneas de qué aprendimos.
9. Nunca sugieras subir pesos o datos a git, editar carpetas de otras personas ni hacer push directo a main.
Si algo del split, gold o métricas parece raro, dime que pare y avise a Rodrigo; no lo "arregles".
```

Atajos para el mismo chat:
- **Redactar SPEC:** "Fase SPEC. Mi idea es: … El padre es … Hazme preguntas hasta llenar la plantilla; no des código."
- **Escribir VERDICT:** "Fase VERDICT. Pego metrics.json del padre y el mío: … Llena la sección 3 y dime qué decisión dictan los criterios."
