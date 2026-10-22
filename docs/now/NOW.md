# NOW — estado vivo del reto RSNA Knee (actualizado 2026-10-19 ~15:10 UTC)

> Léelo entero al empezar cualquier sesión (Claude Code o Codex). Es la fuente de verdad del "ahora".
> Historia detallada: `docs/EXPERIMENTOS.md`, `docs/pasos/envios_kaggle.md`, `docs/pasos/*.md`, y en el volumen
> `/workspace/users/rodrigo/orq/ESTADO.md` + `orq/<agente>/HANDOFF.md`. Si algo aquí contradice esos archivos, gana el
> archivo más reciente; actualiza este NOW.md al cerrar cada bloque de trabajo.
> Codex: lee también `docs/now/CODEX.md` (cómo operar sin Claude).

## 1. Qué es y fechas
- Competencia Kaggle `rsna-knee-abnormality-detection` (12 hallazgos por estudio, métrica AUC macro). Cuenta Kaggle
  `medellinrodrigo` (equipo de Rodrigo). **Cierre: 22-oct-2026 23:59 UTC.** Winners' requirements 5-nov.
- **Entrenamiento congelado desde 19-oct 12:00 UTC** (regla propia): desde ahí solo ensambles/kernels con pesos ya
  entrenados. Envíos: **5 por día** (UTC), nunca otra cuenta.
- Metas: (1) premio del track de eficiencia, (2) paper en revista (después del cierre; no publicar código/preprint antes del 22-oct),
  (3) mejor LB posible.
- Licencia de datos MIRA: nunca redistribuir datos, informes ni etiquetas.

## 2. Finales (hay que elegirlas A MANO en Kaggle → Submissions → Select antes del 22-oct 23:59 UTC)
| Final | Kernel | Ref envío | LB público | Runtime | Nota |
|---|---|---|---|---|---|
| **A (AUC, "tanque")** | `medellinrodrigo/r6flip-repro` v? | 57315883 | **0.958** | ~horas | copia de heliosli blend-mine: R6 compact (heliosli) + flip TTA + Raptor nartaa 384/224 (224 al 0.35) + pesos Raptor por hallazgo (0.5; Medial OA 0.5, Lateral OA 0.2, PF OA 0.6, Synovitis 0.8, Baker's 0.35) |
| **B (eficiencia)** | `medellinrodrigo/exp080-ctrl224-fast` | 57334779 | **0.949** | ~270–296 s estimado | alumno exp074 (Raptor 224 destilado de maestro R7+Raptor) + cabeza MIL reentrenada (exp080) + lectura rápida |
- Si `tank-v6` o `tank-v6soft` (pendientes, ver §4) sacan ≥ 0.959 y > 0.957, reemplazan a A.
- Respaldo de B: `exp074-student224-fast` (0.948).
- Efficiency ≈ −2.13·AUC + runtime_s/32400 (gana el menor) ⇒ **0.001 AUC ≈ 70 s**. En eficiencia manda el AUC: el top 3
  del Efficiency LB está en 0.958–0.965 AUC; con 0.949 andamos ~puesto 20–25. Bajar runtime ya no da (ruido de máquinas Kaggle ±40 s).

## 3. Envíos recientes (LB público)
| Fecha | Kernel | LB | Verdict |
|---|---|---|---|
| 17-oct | r6flip-repro | 0.958 | worked (final A) |
| 17-oct | r6flip-v2 | 0.958 | didnt |
| 17-oct | r7-raptor-exp067flip | 0.957 | didnt |
| 17-oct | exp074-student224 | 0.948 | worked |
| 18-oct | exp080-ctrl224-fast | 0.949 | worked (final B) |
| 18-oct | exp074-student224-fast / sub078-1-bigru224 / exp080-ctrl224 / sub075-1-solo | 0.948 / 0.948 / 0.949 / 0.942 | ver envios_kaggle.md |
| 19-oct | tank-seeds (exp085) | **0.957** | didnt (pedía ≥ 0.959) |
| 19-oct | tank-v6 (exp088), ref 57400180 | **0.957** | didnt (pedía > 0.957 y ≥ 0.959) |
| 19-oct | tank-v6soft (exp089), ref 57400221 | **0.957** | didnt |
| 20-oct | exp090 exploratorio all-data, ref 57455236 | 0.947 | sin criterio registrado |
| 21-oct | exp082 exploratorio (exp080 + nartaa), ref 57459212 | 0.952 | didnt (umbral por runtime 0.975) |
- Actualizado 22-oct 02:50 UTC con Kaggle en vivo (48 envíos). Final A sigue siendo r6flip-repro 0.958; ningún tanque la supera.
- Registrar todo en `docs/pasos/envios_kaggle.md` (rama `exp010_dev_informes`).

## 4. Experimentos de esta última ronda (16–19 oct)
| Exp | Qué | Resultado | PR |
|---|---|---|---|
| exp074 | alumno Raptor 224 destilado (0.5 maestro + 0.5 consenso) | worked f0 0.8987/0.9197; LB 0.948 | — |
| exp075 | SSL DINO | didnt (LB 0.941/0.942) | — |
| exp076 | limpieza etiquetas OOF | didnt | #27 |
| exp077/078 | GRU / Bi-GRU sobre embeddings | didnt | #24/#25 |
| exp079 | compact destilado | didnt | #31 |
| exp080 | control cabeza MIL reentrenada | f0 didnt, LB 0.949 → final B | #29 |
| exp081 | mitad de cortes + 32 ventanas | didnt (pierde 8× lo que ahorra) | #32 |
| exp082 | ensamble exp080 + nartaa 384 mirror (eficiencia) | **INCOMPLETO**: bench `exp082-bench-ens` COMPLETE pero nadie lo leyó; falta Δt, umbral, kernel de envío, CANDIDATE (ver `orq/eff3b/HANDOFF.md`) | rama exp082_ens_nartaa_mirror |
| exp083 | flip TTA a presupuesto constante | didnt (+0.00002) | #33 |
| exp084 | kernel rápido | killed (+27 s, Δprob 1.4e-3) | #35 |
| exp085 | tank_seeds: 4 semillas fine-tune nartaa 384, 100 % train + gold duro | LB 0.957 didnt | #34 |
| exp086 | etiquetas L6 Claude Opus + debate Claude-vs-GPT | worked (etiquetas listas) | #37 |
| exp087 | kernel nice | didnt | #36 |
| exp088 | tank_v6: 4 semillas con consenso_v6 (afilado 0.95/0.05) | LB 0.957 didnt | ver orq/tank_v6 |
| exp089 | tank_v6soft: 2 semillas con consenso_v6soft (continuo) | LB 0.957 didnt | ver orq/tank_v6 |
| exp091 | stress_v1: robustez solo inferencia, 45 checkpoints (paneles S/R/F) | worked: gate exp029 pasa; ningún candidato supera la regla de subida (≥ 20 % de estabilidad); H1 rho −0.17 | PR #38; resultados en `stress_test/RESULTS.md` |
- Etiquetas exp086 (volumen, NUNCA a git): `/workspace/users/rodrigo/orq/labels_claude/consenso_v6.csv`, `consenso_v6soft.csv`.
  Gold macro AUC (informativo): L6 Opus 0.9065, L6 Sonnet 0.9049, consenso4 0.8948, L5 GPT 0.8891, v6soft 0.8994, v6 0.810 (empates por afilado).
  Debate: 95.5 % acuerdo inicial; 2356 celdas en disputa → 1960 acordaron (GPT cedió 87 %), 79 juez, 317 ambiguas. Costo $88.96 Anthropic + $31.62 OpenAI.
- Pesos (volumen): `/workspace/users/rodrigo/kaggle_export/exp085_tank_seeds/`, `exp088_tank_v6/`, `exp089_tank_v6soft/`.
  Datasets Kaggle privados: `rsna-knee-exp085-tank-seeds`, `rsna-knee-exp088-tank-v6`, `rsna-knee-exp089-tank-v6soft`.

## 5. Qué aprendimos (no repetir)
- El tanque 0.958 está saturado: piezas de la misma familia en el slot compact o Raptor salen neutras (0.957) o restan.
  Semillas propias en el slot 384 restaron (0.957).
- Destilación solo ayuda con maestro de otra familia. Flip TTA no ayuda a modelos entrenados con espejo.
- Recortar lectura/resolución cuesta mucho más AUC de lo que ahorra en runtime.
- Postproceso monotónico por hallazgo no cambia el AUC; solo mezclar información entre hallazgos/modelos lo mueve.
- MaViT público (hengck23, 0.935, ~1.5 s/estudio) no nos gana en eficiencia.
- Barrido público 18-oct: nada nuevo ≥ 0.958; única pieza nueva "B-leg" (R6 + soft labels Qwen, vio gold, licencia "other"), no probada.

## 6. Pendientes (en orden)
1. ~~Leer LB de tank-v6 y tank-v6soft~~ (hecho: 0.957 ambos, didnt). Falta verdict en SPEC/EXPERIMENTOS de exp088/exp089 y exp090/exp082; final A = r6flip-repro (0.958).
2. Terminar exp082 (eficiencia) con los envíos que queden: leer bench, fijar umbral LB = 0.949 + ceil(Δt/70000 a milésima) + 0.001 ANTES de enviar, validar kernel, encolar.
3. Del 20 al 22-oct: máx 5 envíos/día, solo ensambles sin entrenar; una pregunta por envío; criterios fijados antes.
4. **Elegir finales A y B a mano** antes del 22-oct 23:59 UTC.
5. Licencias de assets de la final premiada (B usa Raptor nartaa + pesos propios; anotar licencia nartaa).
6. Mergear PRs abiertos #24–#37 (orden: #26 antes de #30; exp074 antes de los que salen de su rama).
7. Seguridad tras el cierre: rotar la clave Anthropic (se pegó en el chat), borrar `/root/.anthropic_env` del pod rsna-cron, revisar OpenAI.
8. Paper: después del 22-oct (revista Q2 no-MDPI; JIIM candidata).

## 7. Infraestructura
- **Repo:** `C:\Users\medro\Documents\_PERSONAL_PROJECTS\RSNA_Challenge` → GitHub `RodMed0709/RSNA_Challenge_MEXICO`. Rama de docs `exp010_dev_informes`.
  Experimentos: rama `expNNN_tag` + PR a main; nada directo a main; commits SIN trailer Co-Authored-By.
- **Secretos:** `.secrets.env` en la raíz del repo (gitignored; nunca imprimir ni subir): KAGGLE_*, RUNPOD_API_KEY, OPENAI_API_KEY,
  ANTHROPIC_API_KEY (no ligada a workspace) + ANTHROPIC_WORKSPACE_ID. Uso: `set -a; . ./.secrets.env; set +a` en bash.
- **Kaggle CLI:** `python -m kaggle.cli ...` tras cargar secretos. Kernels y datasets SIEMPRE privados. 2 sesiones GPU a la vez por cuenta (cola a veces horas).
- **RunPod:** volumen `lu0c3buems` (EU-RO-1) montado en `/workspace`. REST `rest.runpod.io/v1/pods`, saldo vía GraphQL `myself{clientBalance}`.
  Saldo 19-oct ~$35–43. Máx 5 pods. Solo `rsna-cron` (CPU) queda encendido.
  `pods_guard` mata pods "ociosos" (falla con pods nuevos): crear `/workspace/users/rodrigo/kaggle_queue/guard_skip/<pod>` al entrenar.
- **Pod rsna-cron:** `ssh -p 27671 root@213.173.111.86` (salida con `\r` → `tr -d '\r'`). Cron 00:05 UTC corre
  `kaggle_queue/run_queue.py` sobre `queue_<YYYY-MM-DD>.json` (lista de {kernel, version, message}); registro en
  `submissions_done.csv`, logs `log_<fecha>.txt`. Envío manual el mismo día: añadir a la cola de hoy (backup antes) y
  `python3 run_queue.py <fecha> --only-new`.
- **Orquestador de la nube:** `/workspace/users/rodrigo/orq/PROMPT.md` (backups `.bak_*`); revisa `orq/<dir>/CANDIDATE*.json` y encola en el día siguiente por prioridad.
- **Agentes por experimento:** cada uno en `/workspace/users/rodrigo/orq/<dir>/` con `status.log`, `HANDOFF.md`, `CANDIDATE.json`.
  Brief común de agentes: `docs/now/BRIEF_agentes.md`.

## 8. Workers (para gastar poco)
- Prioridad decidida por Rodrigo (19-oct): **1) Codex** (gratis para Anthropic) → **2) Claude por API key** (workspace Default, ~$120) → 3) suscripción solo para hablar/decidir.
- Codex: `codex-worker -TaskFile <spec.md> -Dir <repo>` (YOLO; commitea pero no pushea). Ver `docs/now/CODEX.md`.
- Claude por API key: `bash docs/now/claude_api.sh <prompt-file> [modelo]` (usa `ANTHROPIC_CUSTOM_HEADERS="anthropic-workspace-id: …"`).
- Sesión interactiva facturada a la API: ver `docs/now/CODEX.md` §"Abrir Claude Code con API key".

## 9. Reglas que Rodrigo fijó (vigentes)
- SDD: SPEC antes de código, una variable por experimento, número con `nuevo_experimento` (desde un pod), criterios antes de ver resultados, verdict obligatorio.
- Gold (58 estudios) solo veta… EXCEPTO el tanque desde 18-oct: Rodrigo autorizó entrenar con 100 % de train incluido gold (etiqueta gold dura); ahí decide el LB.
- Nunca subir a git pesos, predicciones, caches, datos, informes ni etiquetas; de `results/` solo `metrics.json`.
- Código/pesos de heliosli nunca a git. Repos/notebooks públicos se pueden usar (públicos = permitidos); cuidar licencia solo en la final premiada.
- Envíos con preguntas de cosas con evidencia, "no nomás al chilazo" (Rodrigo pidió explícitamente probar algunos didnt en LB).
- Hablar con Rodrigo en español mexicano, directo.
