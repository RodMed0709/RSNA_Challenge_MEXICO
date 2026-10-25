# RSNA Knee — reglas para cualquier sesión de Claude (y para humanos)

Trabajamos con **SDD (spec-driven development)**: ningún experimento se programa sin SPEC escrita y confirmada.
Si en esta sesión no has leído los documentos de abajo, léelos ANTES de proponer, editar o correr nada.

## Lee primero, en este orden
0. `docs/now/NOW.md` — estado vivo (finales, envíos, pendientes). Con Codex, también `docs/now/CODEX.md`.
1. `docs/00_EMPIEZA_AQUI.md` — setup del pod y estructura de carpetas.
2. `docs/02_REGLAS_DEL_EQUIPO.md` — git, carpetas y convivencia en el volumen.
3. `docs/06_SDD.md` — ciclo SPEC → PLAN → IMPLEMENT → RUN → VERDICT → PROMOTE/ARCHIVE y reglas de decisión.
4. `docs/EXPERIMENTOS.md` — registro global: referencia actual y lo que ya falló.
5. `docs/07_SPLIT.md`, `docs/04_DATOS.md`, `docs/05_PLAN_DE_MODELOS.md` — split, datos, backlog priorizado.
   `docs/08_LO_QUE_HACEN_OTROS.md` — qué hacen los notebooks públicos que están en ~0.94.
6. La `SPEC.md` del experimento en curso y la de su padre.

## Reglas no negociables
1. **SPEC antes de código.** Sin las secciones 1 y 2 de `SPEC.md` llenas y confirmadas, no se toca el notebook.
2. **Una variable por experimento** contra el padre. Dos cambios = dos experimentos.
3. **Número global único.** Todo experimento nace con `nuevo_experimento("tag", padre="expNNN")`
   (o `python -m src.knee.experiments nuevo tag --padre expNNN`). Nunca inventes ni reutilices un número.
4. **Se entrena en el notebook** `experiments/expNNN_tag/expNNN_tag.ipynb`, por celdas. No se crean `.py` para experimentos.
   `src/knee/` es código compartido: solo cambia por PR aparte que revisa Rodrigo.
5. **El split no se toca.** Todos usan `split_v1_seed42` con `load_split()`. Un split nuevo es `split_v2_*` con su notebook.
6. **Gold (58 estudios, fold = -1) nunca selecciona nada**: ni época, ni checkpoint, ni hiperparámetro.
   Solo veta: gold cae más de 0.02 contra el padre → `killed`. `SELECT_ON = "pseudo"` siempre.
7. **Decide la pseudo AUC.** Screen en fold 0 contra el padre; promete si Δ ≥ +0.005; se confirma con 5 folds (OOF).
8. **Los criterios de la SPEC no cambian después de ver resultados.**
9. **Verdict obligatorio.** Todo experimento termina con la sección 3 de su SPEC llena y su fila en
   `docs/EXPERIMENTOS.md` (`worked`, `didnt` o `killed`), aunque haya fallado. Los números se copian de `metrics.json`.
10. **Solo escribes en tu carpeta** `/workspace/users/<tu_nombre>/`. Las de otros se leen, nunca se editan.
11. **Nunca se suben a git pesos, predicciones, caches ni datos.** De `results/` solo va `metrics.json`.
12. **Git:** rama `expNNN_tag` desde `main` actualizado; PR a `main` al cerrar el verdict, funcione o no.
    Nada directo a `main`.

## Cómo se hace
- Nuevo experimento: celda `nuevo_experimento(...)` → llena la SPEC → cambia solo las celdas del PLAN.
- Notebook con celdas fijas: 0 parámetros · 1 setup · 2 datos y split · 3 modelo · 4 entrenamiento ·
  5 evaluación · 6 guardar `results/expNNN_tag/metrics.json` · 7 verdict.
- Primero `SMOKE = True` (1 época, pocos estudios). Luego `FOLDS = [0]`. Si promete, `FOLDS = [0, 1, 2, 3, 4]`.
- Worktrees locales **dentro** del repo: `git worktree add worktrees/RSNA_wtNNN expNNN_tag` (nunca `../`).
  Papers, visor y demás material local van en `local/`. Ambos ignorados por git.
- Entrenamientos largos: `bash run_notebook.sh experiments/expNNN_tag/expNNN_tag.ipynb` (sigue aunque cierres el navegador).
- Ver todos los resultados del equipo y rutas de pesos: `tabla_experimentos()`.
- Apaga el pod al terminar.

## Si eres Claude trabajando en este repo
- Antes de editar, di en qué fase SDD está el experimento y qué gate falta. Si falta la SPEC, redáctala
  con la persona y **detente** hasta que la confirme.
- No "mejores" nada fuera del PLAN. Si ves algo, anótalo como idea para otro experimento.
- Nunca inventes números: salen de `metrics.json`.

## Si no estás seguro
- ¿Una o dos variables? → dos experimentos.
- ¿Puedo tocar este archivo? Si no está en tu `experiments/expNNN_tag/` → no; pregunta a Rodrigo.
- ¿Algo raro con split, gold o métricas? → para, no lo "arregles", avisa al equipo.
