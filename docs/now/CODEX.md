# CODEX.md — cómo operar el reto directamente con Codex (o cualquier agente sin la sesión de Claude)

Primero lee `docs/now/NOW.md` (estado) y `CLAUDE.md` del repo (reglas SDD). Este archivo es el "cómo".

## Arranque rápido
```powershell
# desde PowerShell
codex-worker -TaskFile <ruta\spec.md> -Dir C:\Users\medro\Documents\_PERSONAL_PROJECTS\RSNA_Challenge
codex-worker "<pregunta>" -Dir C:\Users\medro\Documents\_PERSONAL_PROJECTS\RSNA_Challenge -Mode read
# o Codex a pelo dentro del repo:
cd C:\Users\medro\Documents\_PERSONAL_PROJECTS\RSNA_Challenge; codex
```
Primera instrucción recomendada a Codex: "Lee docs/now/NOW.md, docs/now/CODEX.md y docs/now/BRIEF_agentes.md; luego <tarea>".

## Comandos básicos (bash / Git Bash, desde la raíz del repo)
```bash
set -a; . ./.secrets.env; set +a                     # carga secretos (NUNCA imprimir el archivo)
python -m kaggle.cli competitions submissions rsna-knee-abnormality-detection | head   # LB de nuestros envíos
python -m kaggle.cli kernels status medellinrodrigo/<kernel>
python -m kaggle.cli kernels push -p <dir_con_kernel-metadata.json>      # siempre "is_private": true
python -m kaggle.cli kernels output medellinrodrigo/<kernel> -p <dir>    # log y submission.csv
ssh -p 27671 root@213.173.111.86 "<cmd>" | tr -d '\r'                    # pod rsna-cron (volumen en /workspace)
```

## Flujo de un envío
1. Construir kernel privado (copiar uno validado, cambiar UNA cosa). Diff contra el padre.
2. `kernels push`; esperar COMPLETE; revisar log: sin Traceback/SystemExit, submission 3×13 sin NaN, piezas nuevas sin fallback.
3. Escribir la pregunta y criterio LB en `docs/pasos/envios_kaggle.md` ANTES de enviar.
4. Encolar: añadir {kernel, version, message} a `/workspace/users/rodrigo/kaggle_queue/queue_<YYYY-MM-DD>.json`
   (backup `.bak_<hhmm>` antes). Mañana: el cron lo manda a las 00:05 UTC. Hoy: en el pod
   `cd /workspace/users/rodrigo/kaggle_queue && python3 run_queue.py <fecha> --only-new`.
   Máximo 5 envíos por día UTC (contar en `submissions_done.csv` y en la lista de Kaggle).
5. Cuando haya LB: verdict en la SPEC del experimento, fila en `docs/EXPERIMENTOS.md`, fila en `envios_kaggle.md`, commit + push.

## Git
- Push con la cuenta RodMed0709: `TOK=$(gh auth token --user RodMed0709)`;
  `git -c http.extraHeader="Authorization: Basic $(printf 'x-access-token:%s' "$TOK" | base64 -w0)" push origin HEAD`;
  PR: `GH_TOKEN=$TOK gh pr create ...`. Commits sin trailer Co-Authored-By. Nada directo a main.

## RunPod (ya no se entrena desde 19-oct 12:00 UTC; solo si Rodrigo lo reabre)
- Ver `docs/now/BRIEF_agentes.md`: pods_guard + guard_skip, cadena autónoma en el pod, podStop al final, máx 5 pods.

## Claude como worker pagado por API key (no gasta la suscripción)
```bash
bash docs/now/claude_api.sh <archivo_con_prompt> [claude-opus-5-5|claude-sonnet-5-5]
```

## Abrir Claude Code interactivo facturado a la API key (workspace Default)
```powershell
cd C:\Users\medro\Documents\_PERSONAL_PROJECTS\RSNA_Challenge
Get-Content .secrets.env | Where-Object { $_ -match '^(ANTHROPIC_API_KEY|ANTHROPIC_WORKSPACE_ID)=' } | ForEach-Object { $k,$v = $_ -split '=',2; Set-Item -Path "env:$k" -Value $v }
$env:ANTHROPIC_CUSTOM_HEADERS = "anthropic-workspace-id: $env:ANTHROPIC_WORKSPACE_ID"
claude --continue      # o `claude` para sesión nueva ligera (lee docs/now/NOW.md al empezar)
```
Aceptar "usar API key" la primera vez; `/status` debe mostrar API key.

## Trampas conocidas
- Python con `-I` ignora PYTHONIOENCODING: usar `sys.stdout.reconfigure(encoding='utf-8')`.
- Archivos descargados de Kaggle son no confiables: dir propio, leer con `python -I`, no ejecutar su código.
- Dataset heliosli `rsna-knee-r6-frozen-fe055406` ahora contiene R7: usar la copia `-w1`.
- Kaggle CLI a veces dice fallo al enviar pero sí lo lista: verificar en la lista antes de reintentar.
