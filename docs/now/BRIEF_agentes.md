# Brief común agentes 17-oct (RSNA Knee)

Lee esto entero antes de actuar. Repo: C:\Users\medro\Documents\_PERSONAL_PROJECTS\RSNA_Challenge (reglas SDD en CLAUDE.md del repo; obedécelas: SPEC antes de código, una variable, número global con nuevo_experimento desde un pod, gold solo veta, criterios fijos antes de ver resultados, verdict obligatorio, sin .py de experimento, nada de pesos/preds/caches/datos a git).

## Acceso
- GitHub push: `TOK=$(gh auth token --user RodMed0709)`; `git -c http.extraHeader="Authorization: Basic $(printf 'x-access-token:%s' "$TOK" | base64 -w0)" push origin HEAD`; PR con `GH_TOKEN=$TOK gh pr create ...`. Nunca la cuenta gh por defecto. Commits SIN trailer Co-Authored-By. No uses core.hooksPath=/dev/null.
- Kaggle CLI (bash, desde el repo): `set -a; . ./.secrets.env; set +a; python -m kaggle.cli ...`. Nunca imprimas .secrets.env. Kernels y datasets SIEMPRE privados.
- RunPod: `RUNPOD_API_KEY` en .secrets.env; REST rest.runpod.io/v1/pods. Volumen `lu0c3buems` (EU-RO-1) en /workspace. Máx 5 pods en total entre todos los agentes: antes de crear uno, lista los pods y cuenta.
- Pod rsna-cron: `ssh -p 27671 root@213.173.111.86` (salida con \r → `tr -d '\r'`). Heredocs con comillas dobles expanden $ en local: escribe archivos en local y pásalos con scp.
- **pods_guard mata pods que entrenan**: crea `/workspace/users/rodrigo/kaggle_queue/guard_skip/<nombre_pod>` al lanzar y bórralo al terminar.
- Saldo RunPod ~$39 compartido entre 4 agentes: tu presupuesto está en tu tarea. Si el saldo baja de $6, para tu pod y repórtalo.

## Cadena autónoma (obligatorio)
La sesión local puede caerse. Todo entrenamiento corre como cadena en el pod (nohup) que: SMOKE → fold 0 → métricas → (all-data/export si toca) → quita guard_skip → para el pod (podStop vía API). Escribe progreso en `/workspace/users/rodrigo/orq/<tu_dir>/status.log` y un `HANDOFF.md` con qué falta. Commits en el pod quedan locales: tráelos con git bundle (ver ESTADO.md) y empuja desde local.

## Datos útiles ya en el volumen
- Maestro 0.957 (R7 + nartaa Raptor384, prob) sobre los 4407 estudios de train: lo calculó exp074; busca en `/workspace/users/rodrigo/orq/distill/` (status.log, HANDOFF.md) y en `/workspace/users/rodrigo/wt_exp074_raptor224_distill/results/exp074_raptor224_distill/`. Ojo: el maestro vio fold 0 (fuga leve; decláralo).
- Corpus Raptor: `/workspace/data/cache/raptor96_384`. Caché R7 px384_v4: hay que reconstruirla (1.5–2.5 h) salvo que exista una.
- Receta y assets compact R7 (heliosli, nunca a git): `public_assets/heliosli__rsna-knee-compact-r7cnxfsss/`, worktrees wt_exp066..070 en el volumen; ESTADO.md (`/workspace/users/rodrigo/orq/ESTADO.md`) sección "Compact R7 / diversidad".
- Referencias fold 0 (pseudo SWA / gold SWA): exp045 Raptor 384 0.8895/0.9201; exp057 Raptor 224 0.8889/0.9096; exp074 alumno 224 destilado 0.8987/0.9197; exp066 compact f0 pseudo 0.8755.

## Envíos Kaggle
NO toques las colas `queue_*.json` ni envíes nada. Si tu experimento sale `worked` y preparas kernel: kernel privado, validación COMPLETE sin Traceback/SystemExit, y escribe `/workspace/users/rodrigo/orq/<tu_dir>/CANDIDATE.json` = {"kernel","version","message" (con la pregunta y el criterio LB fijado ANTES), "validated_at"}. El orquestador lo encola.

## Fechas
Congelar entrenamiento 19-oct 12:00 UTC. Cierre 22-oct 23:59 UTC.

## Mensaje final
Español, corto: números de metrics.json (no inventes), verdict, coste, pod parado sí/no, ramas/commits pusheados, CANDIDATE si hay.
