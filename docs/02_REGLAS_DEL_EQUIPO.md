# Reglas del equipo

## Carpetas

Cada persona solo escribe en su carpeta:

```text
/workspace/users/<nombre>/RSNA_Challenge/
/workspace/users/<nombre>/runs/
```

Nunca edites ni borres archivos dentro de `users/<otra_persona>`. Los folders compartidos `data/` y `models/` los actualiza Rodrigo. Todos pueden leerlos.

## Una idea por corrida

Nombra cada run así:

```text
<nombre>_<modelo>_<cambio>_v<N>
```

Ejemplo: `ana_dinov2s_lr1e4_v1`. Cambia una cosa por run para saber qué ayudó. Usa `notes` para describirla.

Al terminar, el entrenamiento agrega automáticamente una fila a `/workspace/results/all_runs.csv`. Si una corrida muere antes, anota el intento manualmente o avisa para no repetirlo.

## Git y notebooks

La opción ordenada es trabajar en una rama personal:

```bash
git switch -c ana/prueba-lr
git add experiments/ana/
git commit -m "experiment: try lower learning rate"
git push -u origin ana/prueba-lr
```

Luego abre un PR hacia `main`. Para cambios pequeños también puedes hacer commit solamente de tus notebooks bajo `experiments/<nombre>/`; no modifiques los originales de `notebooks/` durante una prueba.

No subas datos, caches, pesos, logs ni notebooks ejecutados.

## Uso del pod

- Revisa que no haya otra persona leyendo cientos de GB antes de abrir carpetas grandes.
- No navegues `train_series/` desde JupyterLab mientras alguien entrena.
- Deja entrenamientos largos con `scripts/run_notebook.sh`.
- Apaga el pod cuando termine el entrenamiento. El volumen conserva los archivos; la GPU sigue cobrando mientras el pod está encendido.
