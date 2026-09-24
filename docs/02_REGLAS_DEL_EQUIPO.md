# Reglas del equipo

## Carpetas

- Cada quien trabaja solo en `/workspace/users/<su_nombre>/`. Las carpetas de los demás se pueden leer (por ejemplo,
  para cargar sus pesos), nunca editar.
- `docs/`, `src/`, `splits/` y `kaggle/` son compartidos: cambian por PR, no a mano en tu copia.
- `experiments/expNNN_tag/` es del autor del experimento. `results/expNNN_tag/` guarda lo pesado y no va a git
  (excepto `metrics.json`).
- Nada de carpetas nuevas en `/workspace` ni en `/workspace/data`.

## Experimentos

- Todo experimento sigue `docs/06_SDD.md`: número global → SPEC → PLAN → notebook → fold 0 → verdict.
- Una variable por experimento contra su padre. El padre normalmente es la referencia actual de `docs/EXPERIMENTOS.md`.
- El entrenamiento vive en el notebook del experimento, por celdas. No se crean `.py` para experimentos.
- Si necesitas una función nueva en `src/knee/`, pídela o haz un PR aparte; no la metas dentro de tu experimento.
- El split canónico no se toca. Gold nunca se usa para elegir nada.
- Si un experimento se cae o lo paras, igual escribe el verdict: "killed, se quedó sin memoria con batch 4" también enseña.

## Git

```bash
git switch main && git pull                 # antes de empezar algo nuevo
git switch -c expNNN_tag                    # una rama por experimento
git add experiments/expNNN_tag docs/EXPERIMENTOS.md results/expNNN_tag/metrics.json
git commit -m "expNNN: spec"                # y luego "expNNN: verdict"
git push -u origin HEAD
```

- Al terminar el verdict, abre un PR a `main` en GitHub (el enlace sale al hacer push). Rodrigo lo revisa y lo junta.
- Nunca push directo a `main`. Nunca `git push --force`.
- Los notebooks se suben sin salidas (`nbstripout` las quita solo; lo instala `setup_pod.sh`).
- Si `git pull` da conflicto y no sabes resolverlo: `git merge --abort` y avisa.

## Cómo compartir un hallazgo

1. Tu experimento termina con `decision_auto = promising` en la celda 5.
2. Lo confirmas con `FOLDS = [0, 1, 2, 3, 4]`.
3. Verdict `worked` en la SPEC y en `docs/EXPERIMENTOS.md`, PR a `main`.
4. Rodrigo lo junta y actualiza **Referencia actual** en `docs/EXPERIMENTOS.md`.
5. Avisa en el grupo: "exp023 dinov2-base: OOF 0.84 (+0.01). Ya está en main; úsenlo como padre".
6. Los demás hacen `git pull` y sus siguientes experimentos parten de ahí.

Para usar pesos de otro: `pesos("exp023")` devuelve las rutas de sus `best_f*.pt`. No los copies a tu carpeta.

## Pod y volumen

- Apaga el pod cuando termine tu entrenamiento. El volumen conserva todo; la GPU cobra mientras el pod esté encendido.
- No abras ni recorras `data/raw` (819 mil archivos): vuelve lento el disco para todos. Entrena desde `data/cache`.
- Guarda solo el mejor checkpoint por fold. Revisa espacio con `du -sh /workspace/users/*`.
- Nunca guardes tokens o contraseñas dentro del repo.
