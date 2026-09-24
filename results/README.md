# Resultados

`results/expNNN_tag/` guarda lo que produce el notebook del experimento: `best_f{k}.pt`, `oof_f{k}.csv`, `gold_f{k}.csv`,
`history_f{k}.csv`, logs y `metrics.json`. Solo `metrics.json` se sube a git.

`tabla_experimentos()` junta los `metrics.json` de todas las carpetas `/workspace/users/*/results/` y muestra el ranking
con las rutas de pesos. `pesos("exp023")` devuelve los checkpoints de un experimento.
