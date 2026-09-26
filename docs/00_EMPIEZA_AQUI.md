# Empieza aquí

## Cómo está organizado

```
/workspace/                      volumen compartido de RunPod
├── data/        raw (tal cual de Kaggle) · cache/v1_256 · labels · _logs
├── models/      dinov2-small · Qwen2.5-7B-Instruct
├── venv/        ambiente Python 3.11      venv312/  ambiente Python 3.12
└── users/
    ├── rodrigo/                 clon de GitHub (cada persona tiene el suyo, idénticos)
    └── <tu_nombre>/
        ├── CLAUDE.md            reglas (léelas aunque no uses Claude)
        ├── docs/                compartido: proceso, reglas, registro de experimentos
        ├── setup_pod.sh         prepara el pod (ambiente, kernel, git)
        ├── run_notebook.sh      corre un notebook en segundo plano
        ├── inferencia_kaggle.ipynb   notebook que se sube a Kaggle para enviar
        ├── src/knee/            compartido: código que importan los notebooks (cache, dataset, modelo, métricas, experimentos)
        ├── SPLITS/              compartido: el split canónico (no se edita)
        ├── experiments/         exp001_preparacion (datos, cache, etiquetas, split) · exp001, exp002… en orden
        └── results/             expNNN_tag/ → pesos, predicciones, metrics.json (solo metrics.json va a git)
```

Los números de experimento son **globales**: si Rodrigo va en exp021 y José Javier tomó exp022, el siguiente libre es exp023.
Nadie los elige a mano; los reparte `nuevo_experimento()`.

## Primer día (una sola vez)

1. En RunPod: `Storage` → volumen `RSNA-KNEE` (región `EU-RO-1`) → `Deploy`, template `RSNA-KNEE`,
   GPU de 48 GB o más (ideal `RTX PRO 6000 96GB`).
2. Abre Jupyter → `Terminal` y clona el repo en tu carpeta (Rodrigo te da acceso al repo y te dice cómo crear tu token):
   ```bash
   git clone https://github.com/RodMed0709/RSNA_Challenge_MEXICO.git /workspace/users/<tu_nombre>
   cd /workspace/users/<tu_nombre>
   bash setup_pod.sh
   ```
3. Lee `CLAUDE.md`, `docs/02_REGLAS_DEL_EQUIPO.md` y `docs/06_SDD.md`.

## Cada vez que abras un pod

```bash
cd /workspace/users/<tu_nombre>
bash setup_pod.sh      # elige el ambiente según la versión de Python del pod y registra el kernel
git switch main && git pull
```

En Jupyter usa siempre el kernel **Python (rsna)**.

## Tu primer experimento

1. Abre cualquier notebook y corre una celda:
   ```python
   import sys; sys.path.insert(0, "/workspace/users/<tu_nombre>")
   from src.knee.experiments import nuevo_experimento, tabla_experimentos
   nuevo_experimento("dinov2_base", padre="exp002", idea="backbone base en vez de small")
   ```
   Crea `experiments/expNNN_dinov2_base/` con el notebook del padre y una `SPEC.md` en blanco.
2. Llena las secciones 1 y 2 de la SPEC (puedes pedirle ayuda a Claude con el prompt de `docs/06_SDD.md`).
3. Crea tu rama y sube la SPEC: `git switch -c expNNN_dinov2_base && git add experiments docs && git commit -m "Add expNNN spec" && git push -u origin HEAD`.
4. Cambia solo las celdas del PLAN. Corre el notebook con `SMOKE = True`; si termina bien, `SMOKE = False`, `FOLDS = [0]`.
5. Para dejarlo entrenando aunque cierres el navegador:
   ```bash
   bash run_notebook.sh experiments/expNNN_dinov2_base/expNNN_dinov2_base.ipynb
   ```
6. Llena el verdict (sección 3 de la SPEC y tu fila en `docs/EXPERIMENTOS.md`), commit, push y abre el PR en GitHub.

## Ver cómo va el equipo

```python
tabla_experimentos()        # todos los experimentos de todos, ordenados, con rutas de pesos
```
o en terminal: `python -m src.knee.experiments tabla`.
