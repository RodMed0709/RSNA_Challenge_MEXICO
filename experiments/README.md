# Experimentos personales

Los notebooks de `notebooks/` son la referencia y no se editan para una prueba individual.

Crea tu carpeta y copia ahí el notebook que quieras cambiar:

```bash
mkdir -p experiments/ana
cp notebooks/03_baseline_dinov2.ipynb experiments/ana/ana_dinov2s_lr1e4_v1.ipynb
```

Cada persona trabaja solamente dentro de `experiments/<nombre>/`. Los pesos, logs y notebooks ejecutados van en `/workspace/users/<nombre>/runs/`, no en git.
